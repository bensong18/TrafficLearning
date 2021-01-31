#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# in taxonomic problems"

from distance_builder import *
from distance import *


if __name__ == '__main__':
    builder = DistanceBuilder()
    builder.load_points(r'data/1year_Seattle.txt')
    builder.build_distance_file_for_cluster(EucDistance(), r'data/dis_Seattle.txt')
