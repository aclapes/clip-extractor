====================================
SEGMENT CUTTING SOFTWARE
---
author: aclapes [at] gmail [dot] com
====================================

Description
-----------
This software is intended to extract video cuts that satisfy certain criteria:
(1) Contain one and only face.
(2) Face is not bigger or smaller than a certain size.
(3) Are continuous in time.
(4) Are middle segments (not extrema).


Usage
-----
$ ./videocapture_test.py
    --videos-dir-path=/data/hubpa/Derived/firstimpressions/videos/
    --segments-dir-path=/data/hubpa/Derived/firstimpressions/segments/
    --num_threads=40


Configuration
-------------
Check PARAMETERS dict at the beginning of videocapture_test.py to see
which are the configurable parameters and the default configuration.


Dependences
-----------
- OpenCV 2.4.X
- Some Python libraries
