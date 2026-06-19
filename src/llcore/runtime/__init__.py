# SPDX-License-Identifier: Apache-2.0
"""llcore runtime: run small *pretrained* instruct models locally under llcore's own
memory-efficiency layer (int8 + mmap streaming), so a model too big for comfortable RAM
still converses on a home CPU. Purely additive to the from-scratch char-LM and the
evolutionary/plasticity research substrate — it reuses the int8/mmap primitives, it does
not replace anything.
"""
