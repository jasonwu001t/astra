"""Async tool executor - Async execution support"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry


class AsyncToolExecutor:
    """Async tool executor"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """Execute single tool asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _execute():
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"❌ Tool '{tool_name}' async execution failed: {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in parallel
        
        Args:
            tasks: Task list, each containing tool_name and input_data
            
        Returns:
            Execution result list with task info and results
        """
        print(f"🚀 Starting parallel execution of {len(tasks)} tool tasks")
        
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")
            
            if not tool_name:
                continue
                
            print(f"📝 Creating task {i+1}: {tool_name}")
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, async_task))
        
        results = []
        for i, task, async_task in async_tasks:
            try:
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    "status": "success"
                })
                print(f"✅ Task {i+1} complete: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ Task {i+1} failed: {task['tool_name']} - {e}")
        
        print(f"🎉 Parallel execution complete, success: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        return results

    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        Batch execute same tool
        
        Args:
            tool_name: Tool name
            input_list: Input data list
            
        Returns:
            Execution result list
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """Close executor"""
        self.executor.shutdown(wait=True)
        print("🔒 Async tool executor closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Convenience function: Execute multiple tools in parallel
    
    Args:
        registry: Tool registry
        tasks: Task list
        max_workers: Max worker threads
        
    Returns:
        Execution result list
    """
    executor = AsyncToolExecutor(registry, max_workers)
    try:
        return await executor.execute_tools_parallel(tasks)
    finally:
        executor.close()


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Convenience function: Batch execute same tool
    
    Args:
        registry: Tool registry
        tool_name: Tool name
        input_list: Input data list
        max_workers: Max worker threads
        
    Returns:
        Execution result list
    """
    executor = AsyncToolExecutor(registry, max_workers)
    try:
        return await executor.execute_tools_batch(tool_name, input_list)
    finally:
        executor.close()


# Sync wrapper functions
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Sync version of parallel tool execution"""
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Sync version of batch tool execution"""
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))

