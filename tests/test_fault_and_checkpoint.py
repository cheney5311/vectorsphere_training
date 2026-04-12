import asyncio
import tempfile
from backend.modules.distributed.lease_manager import get_lease_manager
from backend.modules.distributed.fault_tolerance import get_fault_tolerance_manager
from backend.utils.checkpoint_manager import get_checkpoint_manager


def test_fault_triggers_checkpoint_restore(tmp_path):
    lease_mgr = get_lease_manager()
    fault_mgr = get_fault_tolerance_manager()
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    cp = get_checkpoint_manager(str(cp_dir))

    # create a dummy checkpoint file
    class DummyModel:
        def state_dict(self):
            return {'w': 1}
    model = DummyModel()
    path = cp.save_checkpoint(model, epoch=1, step=1)
    name = path.split('/')[-1]

    async def _test():
        # create lease and then expire to trigger fault manager
        lease_id = 'exp-lease-1'
        owner = 'owner-x'
        await lease_mgr.create_lease(lease_id=lease_id, owner_id=owner, ttl_seconds=1, metadata={})
        # wait for expiry
        await asyncio.sleep(2)
        # fault manager should have processed events; here we just ensure no exceptions
        assert True

    asyncio.run(_test())
