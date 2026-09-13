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

"""Regression tests for deterministic contact allocation."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import constraint
from mujoco_warp._src import warp_util


class ContactDeterminismTest(parameterized.TestCase):
  @parameterized.parameters(False, True)
  def test_only_contact_counter_kernels_override_record_default(self, flex):
    """Preserve the caller's dynamic-loop allowance outside contact initialization."""
    if not hasattr(wp, "DeterministicMode"):
      self.skipTest("Requires Warp determinism")
    original_mode = wp.config.deterministic
    original_records = wp.config.deterministic_max_records
    saved_cache = warp_util._KERNEL_CACHE.copy()
    try:
      wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
      wp.config.deterministic_max_records = 32768
      warp_util._KERNEL_CACHE.clear()
      factory = constraint._efc_contact_init_flex if flex else constraint._efc_contact_init
      kernel = factory(mjw.ConeType.ELLIPTIC, True, True)
      self.assertEqual(kernel.module.options["deterministic_max_records"], 0)
      self.assertEqual(kernel.module.options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
      dynamic_kernel = constraint._efc_contact_jac_dense(32, mjw.ConeType.ELLIPTIC)
      self.assertEqual(dynamic_kernel.module.options["deterministic_max_records"], 32768)
    finally:
      wp.config.deterministic = original_mode
      wp.config.deterministic_max_records = original_records
      warp_util._KERNEL_CACHE.clear()
      warp_util._KERNEL_CACHE.update(saved_cache)

  @parameterized.parameters(1, 2, 5)
  def test_contact_capacity_with_large_record_default(self, nworld):
    """Keep contact counters bounded when callers reserve many records for other kernels."""
    if not hasattr(wp, "DeterministicMode") or not wp.is_cuda_available():
      self.skipTest("Requires Warp determinism and CUDA")
    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option cone="elliptic" jacobian="sparse" solver="Newton"/>
        <worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original_mode = wp.config.deterministic
    original_records = wp.config.deterministic_max_records
    saved_cache = warp_util._KERNEL_CACHE.copy()
    try:
      with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=nworld, nconmax=16384, njmax=32768)
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
        wp.config.deterministic_max_records = 32768
        warp_util._KERNEL_CACHE.clear()
        mjw.make_constraint(m, d)
        np.testing.assert_array_equal(d.nefc.numpy(), np.full(nworld, data.nefc))
        addresses = d.contact.efc_address.numpy().copy()
        mjw.make_constraint(m, d)
        np.testing.assert_array_equal(d.contact.efc_address.numpy(), addresses)
        np.testing.assert_array_equal(d.nefc.numpy(), np.full(nworld, data.nefc))
    finally:
      wp.config.deterministic = original_mode
      wp.config.deterministic_max_records = original_records
      warp_util._KERNEL_CACHE.clear()
      warp_util._KERNEL_CACHE.update(saved_cache)


if __name__ == "__main__":
  absltest.main()
