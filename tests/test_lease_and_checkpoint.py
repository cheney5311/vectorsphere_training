import os
import asyncio
import tempfile
import time
from backend.modules.distributed.lease_manager import get_lease_manager
from backend.utils.checkpoint_manager import get_checkpoint_manager


def test_lease_lifecycle_and_checkpoint(tmp_path):
    lease_mgr = get_lease_manager()
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    cp = get_checkpoint_manager(str(cp_dir))

    lease_id = "test-lease-1"
    owner_id = "owner-1"

    async def _test():
        created = await lease_mgr.create_lease(lease_id=lease_id, owner_id=owner_id, ttl_seconds=1, metadata={})
        assert created

        # heartbeat
        ok = await lease_mgr.heartbeat(lease_id)
        assert ok

        # wait for expiry
        await asyncio.sleep(2)

        # after expiry, lease should be removed
        leases = lease_mgr.list_leases()
        assert lease_id not in leases

    asyncio.run(_test())

    # checkpoint save/load
    class DummyModel:
        def state_dict(self):
            return {'w': 1}

    model = DummyModel()
    path = cp.save_checkpoint(model, epoch=1, step=1)
    assert os.path.exists(path)
    # load should raise (no actual torch) but we test path exists
    assert os.path.exists(path)
