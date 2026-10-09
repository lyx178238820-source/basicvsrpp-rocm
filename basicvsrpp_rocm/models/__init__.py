# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

def register_all_modules():
    from basicvsrpp_rocm.models.mmagic import register_all_modules
    register_all_modules()
    from basicvsrpp_rocm.models.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan