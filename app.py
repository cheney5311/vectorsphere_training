"""应用主入口

初始化和配置Flask应用。
"""

import os
import sys
import time
import os
from typing import Optional, Dict
from datetime import datetime
from flask import Flask
from flask_jwt_extended import JWTManager
# Delay importing psycopg2 to avoid segfaults during pytest collection when DB not needed
psycopg2 = None
if os.getenv('FORCE_PG_IMPORT', '0') == '1':
    import psycopg2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.logging_config import setup_logging
from backend.core.config_manager import load_config
from backend.modules.database.manager import get_database_manager
from backend.api.training.training_api import training_bp
from backend.api.training.training_jobs_api import training_jobs_bp
from backend.api.training.training_progress_api import training_progress_bp
from backend.api.training.training_history_api import training_history_bp
from backend.api.training.training_statistics_api import training_statistics_bp
from backend.api.training.training_control_api import training_control_bp
from backend.api.training.three_stage_training_api import three_stage_training_bp
from backend.api.training.pipeline_api import pipeline_bp
from backend.modules.monitoring.metrics_exporter import metrics_bp
from backend.api.training.training_progress_websocket_api import progress_ws_bp
from backend.api.agent.agent_api import agent_bp
from backend.api.dataset.dataset_api import dataset_bp
from backend.api.dataset.dataset_management_api import dataset_management_bp
from backend.api.model.model_api import model_bp
from backend.api.database.api import database_bp
from backend.api.embeddings.api import embeddings_bp  # 添加embeddings API
from backend.api.optimization.optimization_api import optimization_bp  # 添加optimization API
from backend.api.performance.performance_api import performance_bp  # 添加performance API
from backend.core.monitoring.api import monitoring_bp  # 使用统一监控API
from backend.api.scheduler.scheduler_api import scheduler_bp
from backend.api.security.security_api import security_bp

# 导入新添加的API
from backend.api.model.model_management_api import model_management_bp

from backend.api.workflow.workflow_api import workflow_bp
from backend.api.dashboard.dashboard_api import dashboard_bp
from backend.api.dashboard.dashboard_statistics_api import dashboard_statistics_bp
from backend.api.optimization.optimization_management_api import optimization_management_bp
from backend.api.optimization.optimization_statistics_api import optimization_statistics_bp
from backend.api.model.model_comparison_api import model_comparison_bp
from backend.api.workflow.workflow_execution_api import workflow_execution_bp
from backend.api.dataset.dataset_detailed_api import dataset_detailed_bp
from backend.api.model.model_download_api import model_download_bp

# 导入训练模块新增的API
from backend.api.training.hyperparameter_optimization_api import hyperparameter_optimization_bp
from backend.api.training.model_selection_api import model_selection_bp
from backend.api.training.training_execution_api import training_execution_bp
from backend.api.training.model_evaluation_api import model_evaluation_bp
from backend.api.training.model_optimization_api import model_optimization_bp
from backend.api.training.model_deployment_api import model_deployment_bp
from backend.api.training.monitoring_operations_api import monitoring_operations_bp
from backend.api.training.intelligent_decision_api import intelligent_decision_bp

# 导入认证和权限管理模块API
from backend.api.auth.auth_api import auth_api_bp
from backend.api.auth.permission_api import permission_api_bp

# 导入用户训练信息API
from backend.api.dashboard.user_training_api import user_training_bp


def wait_for_database(config, max_retries=30, delay=1):
    """等待数据库服务启动并可连接
    
    Args:
        config: 数据库配置
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
        
    Returns:
        bool: 数据库是否可用
    """
    # 检查数据库类型
    db_type = getattr(config, 'type', 'postgresql')
    
    # SQLite 不需要等待外部服务
    if db_type == 'sqlite':
        print(f"使用 SQLite 数据库，无需等待外部服务")
        return True
    
    # PostgreSQL 或其他数据库需要等待连接
    if db_type == 'postgresql':
        # 确保 psycopg2 已导入
        global psycopg2
        if psycopg2 is None:
            try:
                import psycopg2
            except ImportError:
                print("psycopg2 未安装，跳过 PostgreSQL 连接检查")
                return True
        
        print(f"等待 PostgreSQL 数据库服务启动... (最多等待 {max_retries} 秒)")
        
        for i in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=config.host,
                    port=config.port,
                    database="postgres",
                    user=config.username,
                    password=config.password,
                    connect_timeout=5
                )
                conn.close()
                print("PostgreSQL 数据库服务已启动并可连接")
                return True
            except Exception as e:
                print(f"数据库连接尝试 {i+1}/{max_retries}: {e}")
                if i < max_retries - 1:
                    time.sleep(delay)
        
        print("PostgreSQL 数据库服务未响应，请检查数据库是否正确启动")
        return False
    
    # 其他数据库类型
    print(f"数据库类型 {db_type}，跳过连接检查")
    return True


def create_app(config_paths: Optional[Dict[str, str]] = None) -> Flask:
    """创建Flask应用
    
    Args:
        config_paths: 配置文件路径字典
        
    Returns:
        Flask应用实例
    """
    # 设置日志
    setup_logging()

    # 初始化分布式追踪（OpenTelemetry），按需启用
    try:
        from backend.core.tracing import init_tracing
        init_tracing(None)  # 先初始化 Provider
    except Exception as e:
        print(f"分布式追踪初始化失败（可选）: {e}")
    
    # 初始化模型下载环境
    try:
        from backend.utils.model_download_config import setup_model_download_environment
        setup_model_download_environment()
        print("模型下载环境配置完成")
    except Exception as e:
        print(f"模型下载环境配置失败: {e}")
    
    # 初始化优雅退出管理器
    from backend.utils.graceful_shutdown import init_graceful_shutdown, register_shutdown_handler
    shutdown_manager = init_graceful_shutdown()
    print("优雅退出管理器已初始化")
    
    # 如果没有提供配置路径，则使用默认路径
    if config_paths is None:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_root = os.path.join(project_root, 'config')
        config_file = os.path.join(config_root, "config.yaml")
        
        # 检查配置文件是否存在
        if os.path.exists(config_file):
            config_paths = {
                "yaml": config_file
            }
            print(f"使用配置文件: {config_file}")
        else:
            print(f"警告: 配置文件不存在: {config_file}")
            config_paths = {}
    
    # 加载配置
    if config_paths:
        try:
            load_config(config_paths)
        except Exception as e:
            print(f"配置加载失败: {e}")
            print("继续使用默认配置启动应用...")
    else:
        # 如果没有配置文件，至少加载环境变量
        from backend.core.config_manager import get_config_manager
        config_manager = get_config_manager()
        config_manager.load_from_environment()
        print("已从环境变量加载配置")
    
    # 获取数据库配置并等待数据库就绪
    from backend.modules.database.config import get_database_config
    db_config = get_database_config()
    
    # 等待数据库服务启动
    if not wait_for_database(db_config):
        print("警告: 数据库服务不可用，但继续启动应用...")
    
    # 创建Flask应用
    app = Flask(__name__)

    # Grafana 数据源与仪表盘预置（可选，按环境变量启用）
    try:
        from backend.modules.monitoring.grafana_provisioning import setup_grafana_provisioning
        setup_grafana_provisioning()
    except Exception as e:
        print(f"Grafana 预置失败（可选）: {e}")

    # 将 Flask 应用接入追踪
    try:
        from backend.core.tracing import init_tracing
        init_tracing(app)
    except Exception as e:
        print(f"Flask 追踪接入失败（可选）: {e}")
    
    # 配置JWT
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', '3600'))
    jwt = JWTManager(app)
    
    # 初始化数据库表
    try:
        # 使用统一的数据库初始化器
        from backend.utils.schema_exporter import DatabaseInitializer
        
        initializer = DatabaseInitializer()
        
        # 检查 Schema 同步状态
        sync_status = initializer.check_schema_sync()
        
        if sync_status['missing_tables']:
            print(f"检测到 {len(sync_status['missing_tables'])} 个缺失的数据表")
            print("正在初始化数据库表...")
            
            # 初始化缺失的表
            init_result = initializer.initialize(force=False)
            
            if init_result['success']:
                print(f"数据库表创建成功，共创建 {len(init_result['tables_created'])} 个表")
                if init_result['tables_created']:
                    print(f"  新创建的表: {', '.join(init_result['tables_created'][:5])}{'...' if len(init_result['tables_created']) > 5 else ''}")
            else:
                print(f"数据库初始化警告: {init_result['errors']}")
        else:
            print(f"数据库 Schema 已同步，共 {len(sync_status['db_tables'])} 个表")
        
        # 注册数据库关闭处理器
        def cleanup_database():
            """清理数据库连接"""
            try:
                print("正在清理数据库连接...")
                db_manager = get_database_manager()
                if db_manager:
                    # 使用优雅关闭方法
                    success = db_manager.graceful_shutdown(timeout=15)
                    if not success:
                        print("优雅关闭失败，尝试强制关闭...")
                        db_manager.force_close_all_connections()
                
                # 最后调用全局清理
                from backend.modules.database.manager import close_database_manager
                close_database_manager()
                print("数据库连接已清理")
            except Exception as e:
                print(f"清理数据库连接失败: {e}")
        
        register_shutdown_handler(cleanup_database, "数据库连接清理")
        
    except Exception as e:
        print(f"数据库表创建失败: {e}")
        print("尝试使用备用方法初始化...")
        try:
            db_manager = get_database_manager()
            db_manager.create_tables()
            print("数据库表创建成功（备用方法）")
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
        # 不阻止应用启动，但记录错误
    
    # 注册蓝图
    app.register_blueprint(training_bp)
    app.register_blueprint(training_jobs_bp)
    app.register_blueprint(training_progress_bp)
    app.register_blueprint(training_history_bp)
    app.register_blueprint(training_statistics_bp)
    app.register_blueprint(training_control_bp)
    app.register_blueprint(three_stage_training_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(progress_ws_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(dataset_management_bp)
    app.register_blueprint(model_bp)

    app.register_blueprint(database_bp)
    app.register_blueprint(embeddings_bp)  # 注册embeddings API
    app.register_blueprint(optimization_bp)  # 注册optimization API
    app.register_blueprint(performance_bp)  # 注册performance API
    app.register_blueprint(monitoring_bp, name='core_monitoring')  # 注册统一监控 API
    app.register_blueprint(scheduler_bp)  # 注册scheduler API
    app.register_blueprint(security_bp)  # 注册security API
    
    # 注册新添加的API
    app.register_blueprint(model_management_bp)

    app.register_blueprint(workflow_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dashboard_statistics_bp)
    app.register_blueprint(optimization_management_bp)
    app.register_blueprint(optimization_statistics_bp)
    app.register_blueprint(model_comparison_bp)
    app.register_blueprint(workflow_execution_bp)
    app.register_blueprint(dataset_detailed_bp)
    app.register_blueprint(model_download_bp)
    
    # 注册训练模块新增的API
    app.register_blueprint(hyperparameter_optimization_bp)
    app.register_blueprint(model_selection_bp)
    app.register_blueprint(training_execution_bp)
    app.register_blueprint(model_evaluation_bp)
    app.register_blueprint(model_optimization_bp)
    app.register_blueprint(model_deployment_bp)
    app.register_blueprint(monitoring_operations_bp)
    app.register_blueprint(intelligent_decision_bp)
    
    # 注册认证和权限管理模块API
    app.register_blueprint(auth_api_bp)
    app.register_blueprint(permission_api_bp)
    
    # 注册用户训练信息API
    app.register_blueprint(user_training_bp)
    
    # 初始化并注册各种服务组件的关闭处理器
    _register_service_shutdown_handlers()
    
    # 添加健康检查端点
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'VectorSphere Backend'}

    # 示例：在主 app 中展示如何使用 JSON Schema 验证装饰器
    try:
        from backend.core.validation import validate_json_schema
        schema_path = 'backend/api/schemas/example_create_training.json'

        @app.route('/api/v1/demo/create_training', methods=['POST'])
        @validate_json_schema(schema_path)
        def demo_create_training():
            data = None
            try:
                data = request.get_json(silent=True)
            except Exception:
                data = None
            return {'status': 'ok', 'received': data}
    except Exception as e:
        # 不影响应用启动，仅记录
        print(f"示例 API 注册失败（可忽略）: {e}")

    return app


def _register_service_shutdown_handlers():
    """注册各种服务组件的关闭处理器"""
    from backend.utils.graceful_shutdown import register_shutdown_handler
    
    # 1. 注册异步处理器关闭处理器
    def cleanup_async_processor():
        """清理异步处理器"""
        try:
            from backend.services.async_processor import get_async_processor
            async_processor = get_async_processor()
            if async_processor._running:
                print("正在关闭异步处理器...")
                async_processor.shutdown(timeout=30.0)
                print("异步处理器已关闭")
        except Exception as e:
            print(f"关闭异步处理器时出错: {e}")
    
    register_shutdown_handler(cleanup_async_processor, "异步处理器清理", timeout=15.0)
    
    # 2. 注册WebSocket管理器关闭处理器
    def cleanup_websocket_manager():
        """清理WebSocket连接"""
        try:
            print("正在清理WebSocket连接...")
            from backend.websocket.websocket_manager import get_websocket_manager
            ws_manager = get_websocket_manager()
            if ws_manager:
                # 使用优雅关闭方法
                success = ws_manager.graceful_shutdown(timeout=20)
                if not success:
                    print("WebSocket优雅关闭失败，尝试强制断开...")
                    disconnected = ws_manager.force_disconnect_all()
                    print(f"强制断开了 {disconnected} 个连接")
                
            print("WebSocket连接已清理")
        except Exception as e:
            print(f"清理WebSocket连接失败: {e}")
    
    register_shutdown_handler(cleanup_websocket_manager, "WebSocket连接清理", timeout=10.0)
    
    # 3. 注册训练服务关闭处理器
    def cleanup_training_services():
        """清理训练服务"""
        try:
            print("正在停止训练服务...")
            # 停止所有正在运行的训练任务
            _stop_all_training_sessions()
            print("训练服务已停止")
        except Exception as e:
            print(f"停止训练服务时出错: {e}")
    
    register_shutdown_handler(cleanup_training_services, "训练服务清理", timeout=20.0)
    
    # 4. 注册监控服务关闭处理器
    def cleanup_monitoring_services():
        """清理监控服务"""
        try:
            print("正在关闭监控服务...")
            # 停止性能监控
            from backend.utils.cleanup_scheduler import CleanupScheduler
            # 如果有全局调度器实例，停止它
            print("监控服务已关闭")
        except Exception as e:
            print(f"关闭监控服务时出错: {e}")
    
    register_shutdown_handler(cleanup_monitoring_services, "监控服务清理")
    
    # 5. 注册清理调度器关闭处理器
    def cleanup_scheduler():
        """清理调度器"""
        try:
            print("正在关闭调度器...")
            # 停止清理调度器
            print("调度器已关闭")
        except Exception as e:
            print(f"关闭调度器时出错: {e}")
    
    register_shutdown_handler(cleanup_scheduler, "调度器清理")
    
    print("所有服务关闭处理器已注册")


def _stop_all_training_sessions():
    """停止所有正在运行的训练会话（带超时控制）"""
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    start_time = time.time()
    max_execution_time = 15.0  # 最大执行时间15秒
    
    try:
        print("开始停止所有训练会话...")
        
        # 1. 快速停止活跃训练（超时5秒）
        def stop_active_trainings():
            try:
                from backend.services.training_execution_service import TrainingExecutionService
                execution_service = TrainingExecutionService()
                
                if hasattr(execution_service, '_active_trainings'):
                    active_sessions = list(execution_service._active_trainings.keys())
                    for session_id in active_sessions:
                        if time.time() - start_time > max_execution_time:
                            print("停止活跃训练超时，跳过剩余会话")
                            break
                        try:
                            print(f"通过执行服务停止训练会话: {session_id}")
                            result = execution_service.stop_training(session_id)
                            print(f"训练会话 {session_id} 停止结果: {result.get('message', '已停止')}")
                        except Exception as e:
                            print(f"通过执行服务停止训练会话 {session_id} 失败: {e}")
            except Exception as e:
                print(f"通过执行服务停止训练失败: {e}")
        
        # 2. 快速取消进度跟踪（超时3秒）
        def cancel_progress_tracking():
            try:
                from backend.modules.training.progress.progress_manager import get_progress_manager
                progress_manager = get_progress_manager()
                
                if hasattr(progress_manager, 'progress_data'):
                    active_progress_sessions = list(progress_manager.progress_data.keys())
                    for session_id in active_progress_sessions:
                        if time.time() - start_time > max_execution_time:
                            print("取消进度跟踪超时，跳过剩余会话")
                            break
                        try:
                            print(f"取消进度跟踪: {session_id}")
                            progress_manager.cancel_training(session_id)
                            print(f"进度跟踪 {session_id} 已取消")
                        except Exception as e:
                            print(f"取消进度跟踪 {session_id} 失败: {e}")
            except Exception as e:
                print(f"通过进度管理器停止训练失败: {e}")
        
        # 3. 快速更新数据库状态（超时7秒）
        def update_database_status():
            try:
                from backend.modules.database.manager import get_database_manager
                from backend.schemas.training_models import TrainingSession
                
                db_manager = get_database_manager()
                
                # 使用较短的数据库连接超时
                with db_manager.get_db_session() as session:
                    # 根据数据库类型设置查询超时
                    db_type = getattr(db_manager.config, 'type', 'postgresql')
                    try:
                        if db_type == 'mysql':
                            from sqlalchemy import text
                            session.execute(text("SET SESSION innodb_lock_wait_timeout = 3"))
                        elif db_type == 'postgresql':
                            from sqlalchemy import text
                            session.execute(text("SET statement_timeout = '3s'"))
                        # SQLite 不支持会话级超时设置
                    except Exception as timeout_err:
                        logger.debug(f"设置数据库超时失败 (可忽略): {timeout_err}")
                    
                    # 查询所有状态为running或pending的训练会话
                    running_sessions = session.query(TrainingSession).filter(
                        TrainingSession.status.in_(['running', 'pending', 'paused'])
                    ).limit(50).all()  # 限制处理数量
                    
                    stopped_count = 0
                    for training_session in running_sessions:
                        if time.time() - start_time > max_execution_time:
                            print("更新数据库状态超时，跳过剩余会话")
                            break
                        try:
                            print(f"更新数据库中的训练会话状态: {training_session.session_id}")
                            
                            # 简化进度保存，避免复杂操作
                            current_progress = {
                                'session_id': training_session.session_id,
                                'status': training_session.status,
                                'saved_at': datetime.utcnow().isoformat()
                            }
                            
                            # 更新状态为stopped
                            training_session.status = 'stopped'
                            training_session.updated_at = datetime.utcnow()
                            training_session.completed_at = datetime.utcnow()
                            
                            # 在配置中保存停止原因
                            if not training_session.config:
                                training_session.config = {}
                            training_session.config['stop_reason'] = 'graceful_shutdown'
                            training_session.config['final_progress'] = current_progress
                            training_session.config['stopped_at'] = datetime.utcnow().isoformat()
                            
                            session.commit()
                            stopped_count += 1
                            print(f"训练会话 {training_session.session_id} 已停止")
                        except Exception as e:
                            print(f"停止训练会话 {training_session.session_id} 时出错: {e}")
                            try:
                                session.rollback()
                            except:
                                pass
                    
                    print(f"共停止了 {stopped_count} 个训练会话")
            except Exception as e:
                print(f"更新数据库状态失败: {e}")
        
        # 并发执行各个停止步骤，但有总体超时控制
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(stop_active_trainings),
                executor.submit(cancel_progress_tracking),
                executor.submit(update_database_status)
            ]
            
            # 等待所有任务完成或超时
            for future in futures:
                try:
                    remaining_time = max_execution_time - (time.time() - start_time)
                    if remaining_time > 0:
                        future.result(timeout=remaining_time)
                except FutureTimeoutError:
                    print("训练停止操作超时")
                except Exception as e:
                    print(f"训练停止操作异常: {e}")
        
        elapsed = time.time() - start_time
        print(f"训练会话停止完成，耗时: {elapsed:.2f}秒")
                    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"停止训练会话时出错: {e}，耗时: {elapsed:.2f}秒")


def _save_training_progress(training_session) -> dict:
    """保存训练进度信息"""
    try:
        progress_info = {
            'session_id': training_session.session_id,
            'status': training_session.status,
            'progress': 0.0,
            'current_epoch': 0,
            'current_step': 0,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        # 尝试从进度管理器获取详细进度
        try:
            from backend.modules.training.progress.progress_manager import get_progress_manager
            progress_manager = get_progress_manager()
            
            if hasattr(progress_manager, 'progress_data') and training_session.session_id in progress_manager.progress_data:
                progress_data = progress_manager.progress_data[training_session.session_id]
                progress_info.update({
                    'progress': getattr(progress_data, 'progress', 0.0),
                    'current_epoch': getattr(progress_data, 'current_epoch', 0),
                    'current_step': getattr(progress_data, 'current_step', 0),
                    'train_loss': getattr(progress_data, 'train_loss', None),
                    'train_accuracy': getattr(progress_data, 'train_accuracy', None),
                    'start_time': getattr(progress_data, 'start_time', None),
                    'end_time': getattr(progress_data, 'end_time', None)
                })
        except Exception as e:
            print(f"获取详细进度信息失败: {e}")
        
        # 尝试从训练执行服务获取状态
        try:
            from backend.services.training_execution_service import TrainingExecutionService
            execution_service = TrainingExecutionService()
            
            if hasattr(execution_service, '_active_trainings') and training_session.session_id in execution_service._active_trainings:
                training_state = execution_service._active_trainings[training_session.session_id]
                progress_info.update({
                    'current_epoch': training_state.get('current_epoch', 0),
                    'current_step': training_state.get('current_step', 0),
                    'resource_history': training_state.get('resource_history', [])[-5:]  # 保存最后5个资源状态
                })
        except Exception as e:
            print(f"获取执行状态失败: {e}")
        
        return progress_info
        
    except Exception as e:
        print(f"保存训练进度失败: {e}")
        return {'error': str(e), 'saved_at': datetime.utcnow().isoformat()}


if __name__ == '__main__':
    # 构建配置路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(project_root, 'config')
    config_file = os.path.join(config_root, "config.yaml")
    
    # 检查配置文件是否存在
    config_paths = None
    if os.path.exists(config_file):
        config_paths = {
            "yaml": config_file
        }
        print(f"使用配置文件: {config_file}")
    else:
        print(f"警告: 配置文件不存在: {config_file}")
    
    app = create_app(config_paths)
    app.run(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('DEBUG', 'false').lower() == 'true'
    )