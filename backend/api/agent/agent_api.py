"""智能体API接口

提供智能体相关的RESTful API接口，支持：
- 智能体 CRUD 操作
- 会话管理
- 长期记忆管理
- 智能对话（同步和流式）
- 知识推理
- 工具管理
- 统计查询
"""

import sys
import os
import asyncio
from flask import Blueprint, request, jsonify, Response, stream_with_context
from flask_jwt_extended import jwt_required, get_jwt_identity
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response
from backend.services.agent_service import (
    AgentService, 
    ChatRequest, 
    get_agent_service
)
from backend.repositories.agent_repository import get_agent_repository
from backend.modules.agent.exceptions.agent_exceptions import (
    AgentNotFoundError, 
    AgentValidationError,
    AgentExecutionError
)
from backend.schemas.agent_type import AgentType


# 创建蓝图
agent_bp = Blueprint('agent', __name__, url_prefix='/api/v1/agents')


def get_service() -> AgentService:
    """获取智能体服务实例"""
    return get_agent_service()


# ============================================================================
# 智能体 CRUD API
# ============================================================================

@agent_bp.route('', methods=['POST'])
@jwt_required()
def create_agent():
    """创建智能体
    
    请求体:
        {
            "name": "智能体名称",
            "description": "描述",
            "agent_type": "chat",
            "version": "1.0.0",
            "config": {},
            "capabilities": [],
            "system_prompt": "系统提示词",
            "model_config": {},
            "is_public": false
        }
        
    返回:
        {
            "success": true,
            "data": {智能体信息},
            "message": "操作成功"
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        name = data.get('name')
        if not name:
            return error_response("智能体名称不能为空", 400)
        
        # 解析智能体类型
        agent_type = None
        agent_type_str = data.get('agent_type')
        if agent_type_str:
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                return error_response(f"不支持的智能体类型: {agent_type_str}", 400)
        
        service = get_service()
        agent = service.create_agent(
            user_id=user_id,
            name=name,
            description=data.get('description'),
            version=data.get('version', '1.0.0'),
            config=data.get('config', {}),
            agent_type=agent_type,
            system_prompt=data.get('system_prompt'),
            model_config=data.get('model_config', {}),
            is_public=data.get('is_public', False),
            capabilities=data.get('capabilities', [])
        )
        
        return success_response(agent.to_dict(), "创建智能体成功", 201)
        
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"创建智能体失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>', methods=['GET'])
@jwt_required()
def get_agent(agent_id):
    """获取智能体详情
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "data": {智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        
        # 检查访问权限
        if agent.user_id != user_id and not agent.config.get('is_public', False):
            return error_response("无权访问此智能体", 403)
        
        return success_response(agent.to_dict())
        
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取智能体失败: {str(e)}", 500)


@agent_bp.route('', methods=['GET'])
@jwt_required()
def list_agents():
    """获取智能体列表
    
    查询参数:
        limit: 返回数量限制 (默认50, 最大100)
        offset: 偏移量 (默认0)
        status: 状态过滤 (active, inactive, deleted)
        agent_type: 类型过滤
        is_public: 是否公开过滤
        
    返回:
        {
            "success": true,
            "data": {
                "agents": [{智能体信息}],
                "limit": 50,
                "offset": 0,
                "count": 10
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        agent_type = request.args.get('agent_type')
        is_public = request.args.get('is_public')
        
        if is_public is not None:
            is_public = is_public.lower() == 'true'
        
        service = get_service()
        agents = service.list_agents(
            user_id=user_id,
            status=status,
            agent_type=agent_type,
            is_public=is_public,
            limit=limit,
            offset=offset
        )
        
        return success_response({
            'agents': [a.to_dict() for a in agents],
            'limit': limit,
            'offset': offset,
            'count': len(agents)
        })
        
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取智能体列表失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>', methods=['PUT'])
@jwt_required()
def update_agent(agent_id):
    """更新智能体
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "name": "新名称",
            "description": "新描述",
            "config": {},
            "system_prompt": "新提示词",
            "model_config": {},
            "is_public": true,
            "capabilities": []
        }
        
    返回:
        {
            "success": true,
            "data": {更新后的智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        service = get_service()
        
        # 检查权限
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权修改此智能体", 403)
        
        updated_agent = service.update_agent(
            agent_id=agent_id,
            name=data.get('name'),
            description=data.get('description'),
            config=data.get('config'),
            system_prompt=data.get('system_prompt'),
            model_config=data.get('model_config'),
            is_public=data.get('is_public'),
            capabilities=data.get('capabilities')
        )
        
        return success_response(updated_agent.to_dict(), "更新成功")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"更新智能体失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>', methods=['DELETE'])
@jwt_required()
def delete_agent(agent_id):
    """删除智能体
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "message": "删除成功"
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        # 检查权限
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权删除此智能体", 403)
        
        success = service.delete_agent(agent_id)
        if success:
            return success_response(None, "删除成功", 200)
        else:
            return error_response("删除智能体失败", 500)
            
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"删除智能体失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/activate', methods=['POST'])
@jwt_required()
def activate_agent(agent_id):
    """激活智能体
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "data": {激活后的智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权操作此智能体", 403)
        
        activated_agent = service.activate_agent(agent_id)
        return success_response(activated_agent.to_dict(), "智能体已激活")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"激活智能体失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/deactivate', methods=['POST'])
@jwt_required()
def deactivate_agent(agent_id):
    """停用智能体
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "data": {停用后的智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权操作此智能体", 403)
        
        deactivated_agent = service.deactivate_agent(agent_id)
        return success_response(deactivated_agent.to_dict(), "智能体已停用")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"停用智能体失败: {str(e)}", 500)


# ============================================================================
# 会话管理 API
# ============================================================================

@agent_bp.route('/<agent_id>/sessions', methods=['POST'])
@jwt_required()
def create_session(agent_id):
    """创建对话会话
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "title": "会话标题",
            "context": {}
        }
        
    返回:
        {
            "success": true,
            "data": {会话信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        service = get_service()
        session = service.create_session(
            agent_id=agent_id,
            user_id=user_id,
            title=data.get('title'),
            context=data.get('context')
        )
        
        return success_response(session, "创建会话成功", 201)
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"创建会话失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/sessions', methods=['GET'])
@jwt_required()
def list_sessions(agent_id):
    """获取会话列表
    
    路径参数:
        agent_id: 智能体ID
        
    查询参数:
        limit: 返回数量限制
        offset: 偏移量
        status: 状态过滤
        
    返回:
        {
            "success": true,
            "data": {
                "sessions": [{会话信息}],
                "count": 10
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        
        service = get_service()
        sessions = service.list_sessions(
            user_id=user_id,
            agent_id=agent_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return success_response({
            'sessions': sessions,
            'count': len(sessions)
        })
        
    except Exception as e:
        return error_response(f"获取会话列表失败: {str(e)}", 500)


@agent_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """获取会话详情
    
    路径参数:
        session_id: 会话ID
        
    返回:
        {
            "success": true,
            "data": {会话信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        session = service.get_session(session_id)
        if not session:
            return error_response("会话不存在", 404)
        
        # 检查权限
        if session.get('user_id') != user_id:
            return error_response("无权访问此会话", 403)
        
        return success_response(session)
        
    except Exception as e:
        return error_response(f"获取会话失败: {str(e)}", 500)


@agent_bp.route('/sessions/<session_id>/messages', methods=['GET'])
@jwt_required()
def get_session_messages(session_id):
    """获取会话消息列表
    
    路径参数:
        session_id: 会话ID
        
    查询参数:
        limit: 返回数量限制
        offset: 偏移量
        
    返回:
        {
            "success": true,
            "data": {
                "messages": [{消息信息}],
                "count": 10
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = get_service()
        
        # 检查会话权限
        session = service.get_session(session_id)
        if not session:
            return error_response("会话不存在", 404)
        if session.get('user_id') != user_id:
            return error_response("无权访问此会话", 403)
        
        messages = service.get_session_messages(session_id, limit, offset)
        
        return success_response({
            'messages': messages,
            'count': len(messages)
        })
        
    except Exception as e:
        return error_response(f"获取消息列表失败: {str(e)}", 500)


@agent_bp.route('/sessions/<session_id>', methods=['PUT'])
@jwt_required()
def update_session(session_id):
    """更新会话
    
    路径参数:
        session_id: 会话ID
        
    请求体:
        {
            "title": "新标题",
            "status": "archived",
            "context": {},
            "summary": "会话摘要"
        }
        
    返回:
        {
            "success": true,
            "data": {更新后的会话信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        service = get_service()
        
        # 检查权限
        session = service.get_session(session_id)
        if not session:
            return error_response("会话不存在", 404)
        if session.get('user_id') != user_id:
            return error_response("无权修改此会话", 403)
        
        updated_session = service.update_session(
            session_id=session_id,
            title=data.get('title'),
            status=data.get('status'),
            context=data.get('context'),
            summary=data.get('summary')
        )
        
        return success_response(updated_session, "更新成功")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"更新会话失败: {str(e)}", 500)


@agent_bp.route('/sessions/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    """删除会话
    
    路径参数:
        session_id: 会话ID
        
    返回:
        {
            "success": true,
            "message": "删除成功"
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        # 检查权限
        session = service.get_session(session_id)
        if not session:
            return error_response("会话不存在", 404)
        if session.get('user_id') != user_id:
            return error_response("无权删除此会话", 403)
        
        success = service.delete_session(session_id)
        if success:
            return success_response(None, "删除成功")
        else:
            return error_response("删除会话失败", 500)
            
    except Exception as e:
        return error_response(f"删除会话失败: {str(e)}", 500)


@agent_bp.route('/sessions/<session_id>/archive', methods=['POST'])
@jwt_required()
def archive_session(session_id):
    """归档会话
    
    路径参数:
        session_id: 会话ID
        
    返回:
        {
            "success": true,
            "data": {归档后的会话信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        session = service.get_session(session_id)
        if not session:
            return error_response("会话不存在", 404)
        if session.get('user_id') != user_id:
            return error_response("无权操作此会话", 403)
        
        archived_session = service.archive_session(session_id)
        return success_response(archived_session, "会话已归档")
        
    except Exception as e:
        return error_response(f"归档会话失败: {str(e)}", 500)


# ============================================================================
# 长期记忆管理 API
# ============================================================================

@agent_bp.route('/<agent_id>/memories', methods=['POST'])
@jwt_required()
def add_memory(agent_id):
    """添加长期记忆
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "content": "记忆内容",
            "memory_type": "fact",
            "importance": 0.5,
            "metadata": {},
            "expires_days": 30
        }
        
    返回:
        {
            "success": true,
            "data": {记忆信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        content = data.get('content')
        if not content:
            return error_response("记忆内容不能为空", 400)
        
        service = get_service()
        memory = service.add_memory(
            agent_id=agent_id,
            user_id=user_id,
            content=content,
            memory_type=data.get('memory_type', 'fact'),
            importance=data.get('importance', 0.5),
            metadata=data.get('metadata'),
            expires_days=data.get('expires_days')
        )
        
        return success_response(memory, "添加记忆成功", 201)
        
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"添加记忆失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/memories', methods=['GET'])
@jwt_required()
def get_memories(agent_id):
    """获取长期记忆列表
    
    路径参数:
        agent_id: 智能体ID
        
    查询参数:
        memory_type: 记忆类型过滤
        min_importance: 最小重要性
        limit: 返回数量限制
        
    返回:
        {
            "success": true,
            "data": {
                "memories": [{记忆信息}],
                "count": 10
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        memory_type = request.args.get('memory_type')
        min_importance = request.args.get('min_importance', 0.0, type=float)
        limit = request.args.get('limit', 50, type=int)
        
        service = get_service()
        memories = service.get_memories(
            agent_id=agent_id,
            user_id=user_id,
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit
        )
        
        return success_response({
            'memories': memories,
            'count': len(memories)
        })
        
    except Exception as e:
        return error_response(f"获取记忆列表失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/memories/search', methods=['POST'])
@jwt_required()
def search_memories(agent_id):
    """搜索相关记忆
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "query": "搜索查询",
            "limit": 10
        }
        
    返回:
        {
            "success": true,
            "data": {
                "memories": [{记忆信息}],
                "count": 5
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        query = data.get('query', '')
        limit = data.get('limit', 10)
        
        if not query:
            return error_response("搜索查询不能为空", 400)
        
        service = get_service()
        memories = service.search_relevant_memories(
            agent_id=agent_id,
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        return success_response({
            'memories': memories,
            'count': len(memories)
        })
        
    except Exception as e:
        return error_response(f"搜索记忆失败: {str(e)}", 500)


@agent_bp.route('/memories/<memory_id>', methods=['DELETE'])
@jwt_required()
def delete_memory(memory_id):
    """删除记忆
    
    路径参数:
        memory_id: 记忆ID
        
    返回:
        {
            "success": true,
            "message": "删除成功"
        }
    """
    try:
        service = get_service()
        success = service.delete_memory(memory_id)
        
        if success:
            return success_response(None, "删除成功")
        else:
            return error_response("删除记忆失败", 500)
            
    except Exception as e:
        return error_response(f"删除记忆失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/memories/consolidate', methods=['POST'])
@jwt_required()
def consolidate_memories(agent_id):
    """整合记忆 - 清理过期和低重要性记忆
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "data": {
                "cleaned_count": 5
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        count = service.consolidate_memories(agent_id, user_id)
        
        return success_response({
            'cleaned_count': count
        }, f"已清理 {count} 条记忆")
        
    except Exception as e:
        return error_response(f"整合记忆失败: {str(e)}", 500)


# ============================================================================
# 智能对话 API
# ============================================================================

@agent_bp.route('/<agent_id>/chat', methods=['POST'])
@jwt_required()
def chat(agent_id):
    """智能对话
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "message": "用户消息",
            "session_id": "会话ID (可选)",
            "context": {},
            "use_memory": true,
            "max_tokens": 2048,
            "temperature": 0.7,
            "provider": "local"
        }
        
    返回:
        {
            "success": true,
            "data": {
                "message": "助手响应",
                "session_id": "会话ID",
                "execution_id": "执行ID",
                "tokens_used": 100,
                "latency_ms": 500,
                "memories_used": ["记忆1", "记忆2"]
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        message = data.get('message')
        if not message:
            return error_response("消息内容不能为空", 400)
        
        chat_request = ChatRequest(
            message=message,
            session_id=data.get('session_id'),
            context=data.get('context'),
            use_memory=data.get('use_memory', True),
            max_tokens=data.get('max_tokens', 2048),
            temperature=data.get('temperature', 0.7),
            provider=data.get('provider', 'local'),
            stream=False
        )
        
        service = get_service()
        
        # 运行异步对话
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                service.chat(agent_id, user_id, chat_request)
            )
        finally:
            loop.close()
        
        return success_response({
            'message': response.message,
            'session_id': response.session_id,
            'execution_id': response.execution_id,
            'tokens_used': response.tokens_used,
            'latency_ms': response.latency_ms,
            'memories_used': response.memories_used or []
        })
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except AgentValidationError as e:
        return error_response(str(e), 400)
    except AgentExecutionError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"对话执行失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/chat/stream', methods=['POST'])
@jwt_required()
def chat_stream(agent_id):
    """流式智能对话
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "message": "用户消息",
            "session_id": "会话ID (可选)",
            "use_memory": true,
            "provider": "local"
        }
        
    返回:
        text/event-stream: 流式响应
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        message = data.get('message')
        if not message:
            return error_response("消息内容不能为空", 400)
        
        chat_request = ChatRequest(
            message=message,
            session_id=data.get('session_id'),
            context=data.get('context'),
            use_memory=data.get('use_memory', True),
            provider=data.get('provider', 'local'),
            stream=True
        )
        
        service = get_service()
        
        def generate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def stream_async():
                    async for chunk in service.chat_stream(agent_id, user_id, chat_request):
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                async_gen = stream_async()
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()
            
            yield "data: [DONE]\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        return error_response(f"流式对话失败: {str(e)}", 500)


# ============================================================================
# 能力管理 API
# ============================================================================

@agent_bp.route('/<agent_id>/capabilities', methods=['POST'])
@jwt_required()
def add_capability(agent_id):
    """添加智能体能力
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "capability": "能力名称"
        }
        
    返回:
        {
            "success": true,
            "data": {更新后的智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        capability = data.get('capability')
        if not capability:
            return error_response("能力名称不能为空", 400)
        
        service = get_service()
        
        # 检查权限
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权操作此智能体", 403)
        
        updated_agent = service.add_agent_capability(agent_id, capability)
        return success_response(updated_agent.to_dict(), "添加能力成功")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"添加能力失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/capabilities/<capability>', methods=['DELETE'])
@jwt_required()
def remove_capability(agent_id, capability):
    """移除智能体能力
    
    路径参数:
        agent_id: 智能体ID
        capability: 能力名称
        
    返回:
        {
            "success": true,
            "data": {更新后的智能体信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        # 检查权限
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权操作此智能体", 403)
        
        updated_agent = service.remove_agent_capability(agent_id, capability)
        return success_response(updated_agent.to_dict(), "移除能力成功")
        
    except AgentNotFoundError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"移除能力失败: {str(e)}", 500)


# ============================================================================
# 工具管理 API
# ============================================================================

@agent_bp.route('/<agent_id>/tools', methods=['POST'])
@jwt_required()
def add_tool(agent_id):
    """添加工具
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "name": "工具名称",
            "description": "描述",
            "tool_type": "custom",
            "schema": {},
            "config": {}
        }
        
    返回:
        {
            "success": true,
            "data": {工具信息}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        name = data.get('name')
        if not name:
            return error_response("工具名称不能为空", 400)
        
        service = get_service()
        
        # 检查权限
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id:
            return error_response("无权操作此智能体", 403)
        
        tool = service.add_tool(
            agent_id=agent_id,
            name=name,
            description=data.get('description', ''),
            tool_type=data.get('tool_type', 'custom'),
            schema=data.get('schema'),
            config=data.get('config')
        )
        
        return success_response(tool, "添加工具成功", 201)
        
    except Exception as e:
        return error_response(f"添加工具失败: {str(e)}", 500)


@agent_bp.route('/<agent_id>/tools', methods=['GET'])
@jwt_required()
def get_tools(agent_id):
    """获取智能体工具列表
    
    路径参数:
        agent_id: 智能体ID
        
    查询参数:
        include_global: 是否包含全局工具 (默认true)
        
    返回:
        {
            "success": true,
            "data": {
                "tools": [{工具信息}],
                "count": 5
            }
        }
    """
    try:
        include_global = request.args.get('include_global', 'true').lower() == 'true'
        
        service = get_service()
        tools = service.get_agent_tools(agent_id, include_global)
        
        return success_response({
            'tools': tools,
            'count': len(tools)
        })
        
    except Exception as e:
        return error_response(f"获取工具列表失败: {str(e)}", 500)


# ============================================================================
# 执行任务 API
# ============================================================================

@agent_bp.route('/<agent_id>/execute', methods=['POST'])
@jwt_required()
def execute_task(agent_id):
    """执行智能体任务
    
    路径参数:
        agent_id: 智能体ID
        
    请求体:
        {
            "input_data": {},
            "context": {}
        }
        
    返回:
        {
            "success": true,
            "data": {执行结果}
        }
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        service = get_service()
        
        # 检查智能体
        agent = service.get_agent(agent_id)
        if not agent:
            return error_response("智能体不存在", 404)
        if agent.user_id != user_id and not agent.config.get('is_public', False):
            return error_response("无权访问此智能体", 403)
        
        input_data = data.get('input_data', {})
        context = data.get('context', {})
        
        # 通过实例管理器执行
        from backend.services.agent_instance_manager import agent_instance_manager
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                agent_instance_manager.execute_agent_task(agent_id, input_data, context)
            )
        finally:
            loop.close()
        
        return success_response(result, "任务执行成功")
        
    except Exception as e:
        return error_response(f"执行任务失败: {str(e)}", 500)


# ============================================================================
# 知识推理 API
# ============================================================================

@agent_bp.route('/knowledge/query', methods=['POST'])
@jwt_required()
def query_knowledge():
    """查询知识库
    
    请求体:
        {
            "query": "查询字符串"
        }
        
    返回:
        {
            "success": true,
            "data": {
                "query": "查询字符串",
                "entities": [{实体}],
                "relationships": [{关系}],
                "recommendations": ["推荐1"]
            }
        }
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        
        if not query:
            return error_response("查询字符串不能为空", 400)
        
        service = get_service()
        result = service.query_knowledge(query)
        
        return success_response(result)
        
    except Exception as e:
        return error_response(f"知识查询失败: {str(e)}", 500)


@agent_bp.route('/training/recommendation', methods=['POST'])
@jwt_required()
def get_training_recommendation():
    """获取训练推荐
    
    请求体:
        {
            "data_type": "text",
            "data_size": "medium",
            "task_type": "classification",
            "gpu_memory": 16
        }
        
    返回:
        {
            "success": true,
            "data": {
                "recommendation": {
                    "model": "transformer",
                    "learning_rate": 2e-5,
                    "batch_size": 32
                },
                "confidence": 0.9,
                "reasoning": "推荐理由"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        service = get_service()
        result = service.get_training_recommendation(data)
        
        return success_response(result)
        
    except Exception as e:
        return error_response(f"获取训练推荐失败: {str(e)}", 500)


# ============================================================================
# 统计查询 API
# ============================================================================

@agent_bp.route('/<agent_id>/statistics', methods=['GET'])
@jwt_required()
def get_agent_statistics(agent_id):
    """获取智能体统计信息
    
    路径参数:
        agent_id: 智能体ID
        
    返回:
        {
            "success": true,
            "data": {
                "execution_count": 100,
                "success_rate": 0.95,
                "avg_response_time": 500,
                "total_sessions": 20,
                "active_sessions": 5,
                "total_memories": 50
            }
        }
    """
    try:
        service = get_service()
        statistics = service.get_agent_statistics(agent_id)
        
        if not statistics:
            return error_response("智能体不存在", 404)
        
        return success_response(statistics)
        
    except Exception as e:
        return error_response(f"获取统计信息失败: {str(e)}", 500)


@agent_bp.route('/summary', methods=['GET'])
@jwt_required()
def get_user_summary():
    """获取用户智能体使用摘要
    
    返回:
        {
            "success": true,
            "data": {
                "total_agents": 5,
                "active_agents": 3,
                "total_sessions": 50,
                "total_executions": 200,
                "weekly_executions": 30
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        service = get_service()
        
        summary = service.get_user_agent_summary(user_id)
        return success_response(summary)
        
    except Exception as e:
        return error_response(f"获取使用摘要失败: {str(e)}", 500)


# ============================================================================
# 搜索和公开智能体 API
# ============================================================================

@agent_bp.route('/search', methods=['GET'])
@jwt_required()
def search_agents():
    """搜索智能体
    
    查询参数:
        q: 搜索关键词
        include_public: 是否包含公开智能体 (默认true)
        limit: 返回数量限制 (默认50)
        
    返回:
        {
            "success": true,
            "data": {
                "agents": [{智能体信息}],
                "count": 10
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        
        query = request.args.get('q', '')
        include_public = request.args.get('include_public', 'true').lower() == 'true'
        limit = request.args.get('limit', 50, type=int)
        
        if not query:
            return error_response("搜索关键词不能为空", 400)
        
        service = get_service()
        agents = service.search_agents(
            query=query,
            user_id=user_id,
            include_public=include_public,
            limit=limit
        )
        
        return success_response({
            'agents': [a.to_dict() for a in agents],
            'count': len(agents)
        })
        
    except Exception as e:
        return error_response(f"搜索智能体失败: {str(e)}", 500)


@agent_bp.route('/public', methods=['GET'])
@jwt_required()
def get_public_agents():
    """获取公开智能体列表
    
    查询参数:
        agent_type: 类型过滤
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    返回:
        {
            "success": true,
            "data": {
                "agents": [{智能体信息}],
                "count": 10
            }
        }
    """
    try:
        agent_type = request.args.get('agent_type')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = get_service()
        agents = service.get_public_agents(
            agent_type=agent_type,
            limit=limit,
            offset=offset
        )
        
        return success_response({
            'agents': [a.to_dict() for a in agents],
            'count': len(agents)
        })
        
    except Exception as e:
        return error_response(f"获取公开智能体失败: {str(e)}", 500)


@agent_bp.route('/types', methods=['GET'])
@jwt_required()
def list_agent_types():
    """获取支持的智能体类型列表
    
    返回:
        {
            "success": true,
            "data": {
                "agent_types": ["chat", "training_assistant", ...]
            }
        }
    """
    try:
        agent_types = [agent_type.value for agent_type in AgentType]
        
        return success_response({
            "agent_types": agent_types
        }, "获取智能体类型列表成功")
        
    except Exception as e:
        return error_response(f"获取智能体类型列表失败: {str(e)}", 500)


# ============================================================================
# 健康检查 API
# ============================================================================

@agent_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    返回:
        {
            "success": true,
            "data": {
                "status": "healthy",
                "service": "agent_api",
                "version": "1.0.0"
            }
        }
    """
    return success_response({
        'status': 'healthy',
        'service': 'agent_api',
        'version': '1.0.0'
    })
