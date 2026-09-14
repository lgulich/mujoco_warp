# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Deterministic sparse Hessian allocation and grid-stride regression tests."""

from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import collision_convex
from mujoco_warp._src import solver
from mujoco_warp._src import types
from mujoco_warp._src import warp_util


class SparseScatterTest(parameterized.TestCase):
  @parameterized.parameters(1, 2, 8193)
  def test_dense_hessian_grid_stride(self, stride):
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    self._with_determinism(lambda: self._check_dense_grid_stride(stride))

  def _check_dense_grid_stride(self, stride):
    ncon = 8193
    with wp.ScopedDevice("cuda:0"):

      def floats(values):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=float)

      def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=int)

      h = wp.zeros((1, 1, 1), dtype=float)
      blocks = (ncon + stride - 1) // stride
      wp.launch(
        solver._update_gradient_JTCJ_dense(blocks),
        dim=(stride, 1),
        inputs=[
          floats([1]),
          ints([0]),
          ints([0]),
          floats([-1] * ncon),
          floats([0] * ncon),
          wp.array(np.ones((ncon, 5), dtype=np.float32), dtype=types.vec5),
          ints([3] * ncon),
          ints([[0, 1, 2, -1, -1, -1]] * ncon),
          ints([0] * ncon),
          floats([[[1], [0], [1]]]),
          floats([[1, 1, 1]]),
          ints([[int(types.ConstraintState.CONE)] * 3]),
          ncon,
          ints([ncon]),
          floats([[0, 1, 0]]),
          wp.array([False], dtype=bool),
          blocks,
          stride,
        ],
        outputs=[h],
      )
      # Each active cone contributes one to this Hessian entry.
      np.testing.assert_array_equal(h.numpy(), [[[float(ncon)]]])

  @parameterized.parameters(
    (1, 8192, 8192, 33, False),
    *((worlds, 17, 3, 33, compact) for worlds in (1, 2, 5) for compact in (False, True)),
  )
  def test_sparse_hessian(self, worlds, rows, groups, width, compact):
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    self._with_determinism(lambda: self._check(worlds, rows, groups, width, compact))

  def _with_determinism(self, check):
    mode = wp.config.deterministic
    records = wp.config.deterministic_max_records
    cache = warp_util._KERNEL_CACHE.copy()
    try:
      wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
      wp.config.deterministic_max_records = 8192
      warp_util._KERNEL_CACHE.clear()
      check()
    finally:
      wp.config.deterministic = mode
      wp.config.deterministic_max_records = records
      warp_util._KERNEL_CACHE.clear()
      warp_util._KERNEL_CACHE.update(cache)

  @parameterized.parameters(1, 2, 5)
  def test_solver_capture_matches_eager(self, worlds):
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    self._with_determinism(lambda: self._check_capture(worlds))

  @parameterized.parameters(*((worlds, convex) for worlds in (1, 2, 5) for convex in (False, True)))
  def test_full_capture_matches_eager(self, worlds, convex):
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    self._with_determinism(lambda: self._check_capture(worlds, full=True, convex=convex))

  def test_convex_grid_stride_capture(self):
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    # Five worlds' pairs are deliberately processed by a single thread.
    with mock.patch.object(collision_convex, "_ccd_grid_size", return_value=1):
      self._with_determinism(lambda: self._check_capture(5, full=True, convex=True))

  def _check_capture(self, worlds, full=False, convex=False):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option cone="elliptic" jacobian="sparse" solver="Newton"/>
        <worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody>
      </mujoco>
    """)
    if convex:
      model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option cone="elliptic" jacobian="sparse" solver="Newton"><flag multiccd="enable"/></option>
          <asset><mesh name="cube" vertex="-.1 -.1 -.1 -.1 -.1 .1 -.1 .1 -.1 -.1 .1 .1
                                            .1 -.1 -.1 .1 -.1 .1 .1 .1 -.1 .1 .1 .1"/></asset>
          <worldbody>
            <geom type="box" size="1 1 .05"/>
            <body pos="0 0 .14"><freejoint/><geom type="mesh" mesh="cube"/></body>
          </worldbody>
        </mujoco>
      """)
    if convex:
      self.assertEqual(
        collision_convex._pack_convex_contacts().module.options["deterministic"], wp.DeterministicMode.RUN_TO_RUN
      )
    advance = mjw.step if full else mjw.solve
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice("cuda:0"):
      m = mjw.put_model(model)
      eager = mjw.put_data(model, data, nworld=worlds, nconmax=4096, njmax=8192)
      captured = mjw.put_data(model, data, nworld=worlds, nconmax=4096, njmax=8192)
      # Warm all kernels before capture, leaving both states at the same step.
      advance(m, eager)
      advance(m, captured)
      if full and convex:
        count = int(captured.nacon.numpy()[0])
        self.assertGreaterEqual(count, 2 * worlds)
        self.assertEqual(count, data.ncon * worlds)
        np.testing.assert_allclose(
          np.sort(captured.contact.dist.numpy()[:count]),
          np.sort(np.tile(data.contact.dist[: data.ncon], worlds)),
          atol=1e-5,
          rtol=1e-5,
        )
      with wp.ScopedCapture() as capture:
        advance(m, captured)
      for _ in range(2):
        advance(m, eager)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(captured.qacc.numpy(), eager.qacc.numpy())
        np.testing.assert_array_equal(captured.qfrc_constraint.numpy(), eager.qfrc_constraint.numpy())
        self.assertTrue(np.isfinite(captured.qacc.numpy()).all())
        np.testing.assert_array_equal(captured.qpos.numpy(), eager.qpos.numpy())
        np.testing.assert_array_equal(captured.qvel.numpy(), eager.qvel.numpy())
        np.testing.assert_array_equal(captured.nacon.numpy(), eager.nacon.numpy())
      print(
        f"capture worlds={worlds} full={full} convex={convex} pool_peak={wp.get_mempool_used_mem_high('cuda:0')}", flush=True
      )

  def _check(self, worlds, rows, groups, width, compact):
    def arr(x, dtype):
      return wp.array(np.asarray(x), dtype=dtype, device="cuda:0")

    adr = arr(np.tile(np.arange(rows), (worlds, 1)), wp.int32)
    nrow = arr(np.ones((worlds, rows)), wp.int32)
    count = arr(np.full(worlds, rows), wp.int32)
    nnz = arr(np.full((worlds, rows), width), wp.int32)
    rowadr = arr(np.tile(np.arange(rows) * width, (worlds, 1)), wp.int32)
    cols = arr(np.tile(np.arange(width), (worlds, 1, rows)), wp.int32)
    jac = arr(np.ones((worlds, 1, rows * width)), wp.float32)
    diag = arr(np.ones((worlds, rows)), wp.float32)
    state = arr(np.full((worlds, rows), types.ConstraintState.QUADRATIC.value), wp.int32)
    mapping = arr(np.tile(np.arange(width), (worlds, 1)), wp.int32)
    done = arr(np.zeros(worlds), wp.bool)
    out = wp.zeros((worlds, width, width), dtype=wp.float32, device="cuda:0")
    records = solver._jtdaj_max_records(rows, width, groups)
    kernel = solver._JTDAJ_sparse(compact, records)
    args = [adr, nrow, count, nnz, rowadr, cols, jac, diag, state, mapping, done, groups]
    previous = None
    for repeat in range(2):
      out.zero_()
      wp.launch(kernel, dim=(worlds, groups, 32), inputs=args, outputs=[out], device="cuda:0")
      actual = out.numpy()
      expected = np.broadcast_to(np.triu(np.full((width, width), rows, dtype=np.float32)), actual.shape)
      np.testing.assert_array_equal(actual, expected)
      if previous is not None:
        np.testing.assert_array_equal(actual, previous)
      previous = actual


if __name__ == "__main__":
  absltest.main()
