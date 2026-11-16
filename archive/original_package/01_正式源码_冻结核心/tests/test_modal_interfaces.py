"""Input-interface sanity tests. These are NOT perception/quality benchmarks."""
from dataclasses import replace
from pathlib import Path
import hashlib
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dno import DNO
from examples.units import image_units,video_units,audio_units,vector_units
from examples.run_units import demo

EXPECTED_SHA='fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1'


def test_original_core_still_identical():
    assert hashlib.sha256((ROOT/'dno.py').read_bytes()).hexdigest()==EXPECTED_SHA


@pytest.mark.parametrize('name,got,expect',[
    ('gray', lambda:image_units(torch.arange(64).reshape(8,8)),(4,8)),
    ('rgb', lambda:image_units(torch.zeros(8,8,3)),(4,8)),
    ('video', lambda:video_units(torch.zeros(4,8,8)),(8,8)),
    ('video_rgb', lambda:video_units(torch.zeros(4,8,8,3)),(8,8)),
    ('audio', lambda:audio_units(torch.zeros(256)),(4,8)),
    ('vectors',lambda:vector_units(torch.ones(3,8)),(3,8)),
])
def test_external_shapes(name,got,expect):
    x=got()
    assert x.shape==expect and bool(torch.isfinite(x).all()),name


def test_source_image_position_is_external_metadata():
    # The DNO state stores carrier identities and embeddings, not H,W positions.
    img=image_units(torch.arange(64.).reshape(8,8))
    m=DNO(d=8,hidden=8)
    s=m.introduce(m.blank(),img)
    assert s.value.shape==(4,8) and s.links==()
    assert not hasattr(s,'coordinates')


def test_video_time_is_not_an_automatic_contact():
    torch.manual_seed(1)
    m=DNO(d=8,hidden=8)
    s=m.introduce(m.blank(),video_units(torch.rand(4,8,8)))
    out=m.evolve(s,.1)
    assert out.links==() and out.ids==s.ids


def test_audio_local_input_affects_addressed_identity_only():
    torch.manual_seed(3)
    m=DNO(d=8,hidden=8)
    s=m.introduce(m.blank(),audio_units(torch.randn(256)))
    cue={s.ids[1]:torch.ones(8)}
    a=m.evolve(s,.1)
    b=m.evolve(s,.1,local=cue)
    assert not torch.equal(a.value[1],b.value[1])
    assert torch.equal(a.value[[0,2,3]],b.value[[0,2,3]])


def test_unobserved_unit_remains_locally_independent():
    torch.manual_seed(4)
    m=DNO(d=8,hidden=8)
    s=m.introduce(m.blank(),image_units(torch.rand(8,8)))
    other=replace(s,value=torch.cat((s.value[:3],s.value[3:]+100),0))
    a=m.evolve(s,.1)
    b=m.evolve(other,.1)
    assert torch.equal(a.value[:3],b.value[:3])


def test_joint_examples_do_not_claim_accuracy():
    r=demo()
    assert len(r['results'])==4
    for v in r['results'].values():
        assert v['finite'] and not v['actual_interaction_committed']
        assert abs(v['proposed_source_probability_sum']-1)<1e-5
        assert abs(v['proposed_target_probability_sum']-1)<1e-5


@pytest.mark.parametrize('fn,val',[
    (image_units,torch.zeros(7,8)),
    (video_units,torch.zeros(3,8,8)),
    (audio_units,torch.zeros(250)),
    (vector_units,torch.zeros(3,7)),
])
def test_bad_shapes_explicit(fn,val):
    with pytest.raises(ValueError):fn(val)
