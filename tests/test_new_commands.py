"""测试新重构的commands模块"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_new_commands():
    """测试新commands模块"""
    print("开始测试新重构的commands模块...")
    
    try:
        # 导入新模块
        from backend_new.commands import (
            BaseCommand,
            ServeCommand,
            InfoCommand,
            TrainCommand,
            TestCommand,
            CommandManager
        )
        
        print("✓ 模块导入成功")
        
        # 测试命令管理器
        manager = CommandManager()
        print("✓ 命令管理器创建成功")
        
        # 测试命令注册
        commands = manager.list_commands()
        print(f"✓ 已注册命令: {commands}")
        
        # 测试各个命令实例化
        serve_cmd = ServeCommand()
        info_cmd = InfoCommand()
        train_cmd = TrainCommand()
        test_cmd = TestCommand()
        
        print("✓ 所有命令实例化成功")
        print(f"  - Serve命令: {serve_cmd.name} - {serve_cmd.help}")
        print(f"  - Info命令: {info_cmd.name} - {info_cmd.help}")
        print(f"  - Train命令: {train_cmd.name} - {train_cmd.help}")
        print(f"  - Test命令: {test_cmd.name} - {test_cmd.help}")
        
        # 测试info命令执行
        import argparse
        args = argparse.Namespace(format='text', verbose=False)
        result = info_cmd.execute(args)
        print(f"✓ Info命令执行结果: {result}")
        
        print("\n🎉 新commands模块测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 新commands模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_new_commands()
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败!")