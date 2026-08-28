"""Tests for Vast.ai Cloud provider."""

import pytest

from sky.provision.vast.utils import _create_search_offers_query


def test_search_offers_valid_query():
    got = _create_search_offers_query(instance_type='1x-RTX_5090-32-65536',
                                      region=', CA, NA',
                                      disk_size=32,
                                      secure_only=True)

    assert got == ('chunked=true georegion=true geolocation=NA '
                   'disk_space>=32 num_gpus=1 gpu_name=RTX_5090 '
                   'cpu_ram>=64 datacenter=true')


def test_search_offers_query_survives_the_sdk_parser():
    """The query string is only right if the SDK can read all of it.

    Its parser stops at the first character it cannot match and keeps what it
    parsed so far, so a term it chokes on deletes itself and every term after
    it, without raising. Asserting on the string cannot see that; assert on
    what the SDK turns the string into.
    """
    vast_utils = pytest.importorskip('vastai.utils')
    vast_query = pytest.importorskip('vastai.api.query')

    query_str = _create_search_offers_query(
        instance_type='1x-RTX_5090-32-65536',
        region=', CA, NA',
        disk_size=32,
        secure_only=True)
    _, _, preprocessed = vast_utils.preprocess_search_query(query_str)
    parsed = vast_query.parse_query(preprocessed, {}, vast_query.offers_fields,
                                    vast_query.offers_alias,
                                    vast_query.offers_mult)

    assert set(parsed) == {
        'geolocation', 'disk_space', 'num_gpus', 'gpu_name', 'cpu_ram',
        'datacenter'
    }
    # The region code expands to its countries, and the underscores in the
    # instance type's GPU name come back out as spaces.
    assert 'US' in parsed['geolocation']['in']
    assert parsed['gpu_name'] == {'eq': 'RTX 5090'}
    assert parsed['datacenter'] == {'eq': True}
