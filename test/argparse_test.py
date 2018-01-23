#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

import pytest

from sdrun.main import create_parser

parser = create_parser()

FUNCS = [
    ('prod', ['infile']),
    ('equil', ['infile', 'outfile']),
    ('create', ['outfile']),
]


@pytest.mark.parametrize('func, extras', FUNCS)
def test_verbose(func, extras):
    args = parser.parse_args([func, '-v'] + extras)
    assert args.verbose == 1
    args = parser.parse_args([func, '--verbose'] + extras)
    assert args.verbose == 1
    args = parser.parse_args([func] + ['-v']*3 + extras)
    assert args.verbose == 3


@pytest.mark.parametrize('func, extras', FUNCS)
def test_version(func, extras):
    with pytest.raises(SystemExit) as e:
        parser.parse_args(['--version'])
        assert e == 0
    with pytest.raises(SystemExit) as e:
        parser.parse_args([func, '--version'])
        assert e == 0
    with pytest.raises(SystemExit) as e:
        parser.parse_args(['--version', func])
        assert e == 0


def test_hoomd_args():
    args = parser.parse_args(['equil', '--hoomd-args', '"-mode=cpu"', 'infile', 'outfile'])
    assert args.hoomd_args == '"-mode=cpu"'
