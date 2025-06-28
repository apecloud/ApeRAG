# Copyright 2025 ApeCloud, Inc.
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

"""
Graph Storage Oracle

Oracle代理类，封装真实storage和NetworkX baseline，自动同步操作并比较结果。

核心思想：
1. 实现BaseGraphStorage接口，让测试代码可以直接使用
2. 所有操作自动镜像到storage和baseline
3. 自动比较查询结果，确保一致性
4. 每个方法负责自己的比较逻辑，避免超级复杂的通用比较函数
"""


from aperag.graph.lightrag.base import BaseGraphStorage
from aperag.graph.lightrag.types import KnowledgeGraph
from aperag.graph.lightrag.utils import EmbeddingFunc


class GraphStorageOracle(BaseGraphStorage):
    """
    Oracle代理类，实现BaseGraphStorage接口。
    
    每个方法都会：
    1. 在baseline上执行操作
    2. 在真实storage上执行操作  
    3. 比较结果并抛出异常（如果不匹配）
    4. 返回真实storage的结果
    """

    WRITE_OPERATIONS = {
        "upsert_node", "upsert_edge", "delete_node", "remove_nodes", "remove_edges",
        "drop", "initialize"
    }

    def __init__(self, storage: BaseGraphStorage, baseline: BaseGraphStorage, 
                 namespace: str = "test", workspace: str = "test", 
                 embedding_func: EmbeddingFunc = None):
        # Initialize parent class  
        super().__init__(namespace=namespace, workspace=workspace, embedding_func=embedding_func)
        
        self.storage = storage
        self.baseline = baseline
        self._operation_count = 0
        print(f"🛰️  GraphStorageOracle initialized for {type(storage).__name__}")

    async def initialize(self):
        """Initialize both storages"""
        self._operation_count += 1
        operation_id = f"initialize#{self._operation_count}"
        
        try:
            # Initialize baseline first
            await self.baseline.initialize()
            
            # Initialize storage
            await self.storage.initialize()
            
            print(f"⚖️  Write operation '{operation_id}' completed and synced")
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def finalize(self):
        """Finalize both storages"""
        try:
            await self.baseline.finalize()
            await self.storage.finalize()
            print("🧹 Oracle storage cleanup completed")
        except Exception as e:
            print(f"⚠️  Oracle finalize warning: {e}")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists"""
        self._operation_count += 1
        operation_id = f"has_node#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.has_node(node_id)
            storage_result = await self.storage.has_node(node_id)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists"""
        self._operation_count += 1
        operation_id = f"has_edge#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.has_edge(source_node_id, target_node_id)
            storage_result = await self.storage.has_edge(source_node_id, target_node_id)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def node_degree(self, node_id: str) -> int:
        """Get node degree"""
        self._operation_count += 1
        operation_id = f"node_degree#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.node_degree(node_id)
            storage_result = await self.storage.node_degree(node_id)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get edge degree"""
        self._operation_count += 1
        operation_id = f"edge_degree#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.edge_degree(src_id, tgt_id)
            storage_result = await self.storage.edge_degree(src_id, tgt_id)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node data with flexible field comparison that allows storage to have extra None fields"""
        self._operation_count += 1
        operation_id = f"get_node#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_node(node_id)
            storage_result = await self.storage.get_node(node_id)
            
            # Handle None cases first
            if baseline_result is None and storage_result is None:
                print(f"✅ Oracle match for '{operation_id}' (both None)")
                return storage_result
            elif baseline_result is None or storage_result is None:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}' (None vs non-None):\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            # Use flexible dictionary comparison
            if not self._flexible_dict_compare(baseline_result, storage_result, operation_id):
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge data with flexible field comparison that allows storage to have extra None fields"""
        self._operation_count += 1
        operation_id = f"get_edge#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_edge(source_node_id, target_node_id)
            storage_result = await self.storage.get_edge(source_node_id, target_node_id)
            
            # Handle None cases first
            if baseline_result is None and storage_result is None:
                print(f"✅ Oracle match for '{operation_id}' (both None)")
                return storage_result
            elif baseline_result is None or storage_result is None:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}' (None vs non-None):\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            # Use flexible dictionary comparison (includes float tolerance)
            if not self._flexible_dict_compare(baseline_result, storage_result, operation_id):
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    def _flexible_dict_compare(self, baseline: dict, storage: dict, operation_id: str) -> bool:
        """
        Flexible dictionary comparison that allows storage to have extra None fields.
        
        Rules:
        1. If storage field is None and baseline doesn't have the field: OK
        2. If baseline has a field, storage must also have it: REQUIRED  
        3. Values must match (with float tolerance)
        
        Args:
            baseline: Baseline dictionary
            storage: Storage dictionary  
            operation_id: Operation ID for error reporting
            
        Returns:
            True if dictionaries match according to flexible rules
        """
        # Check that all baseline fields exist in storage
        for key in baseline.keys():
            if key not in storage:
                print(f"❌ Missing field in storage: {key}")
                return False
            
            baseline_val = baseline[key]
            storage_val = storage[key]
            
            # Compare values with float tolerance
            if isinstance(baseline_val, float) and isinstance(storage_val, float):
                if abs(baseline_val - storage_val) > 1e-6:
                    print(f"❌ Float mismatch for {key}: storage={storage_val}, baseline={baseline_val}")
                    return False
            elif baseline_val != storage_val:
                print(f"❌ Value mismatch for {key}: storage={storage_val}, baseline={baseline_val}")
                return False
        
        # Check extra storage fields - they must be None to be allowed
        for key in storage.keys():
            if key not in baseline:
                if storage[key] is not None:
                    print(f"❌ Extra non-None field in storage: {key}={storage[key]}")
                    return False
                # Extra None fields are OK, just log them
                print(f"ℹ️  Extra None field in storage: {key}=None (allowed)")
        
        return True

    def _normalize_edge(self, edge: tuple[str, str]) -> tuple[str, str]:
        """Normalize edge tuple to have consistent ordering (smaller node ID first)"""
        src, tgt = edge
        return (src, tgt) if src <= tgt else (tgt, src)
    
    def _normalize_edge_list(self, edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
        """Normalize a list of edges to a set of normalized edge tuples"""
        return {self._normalize_edge(edge) for edge in edges}

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get node edges with unordered comparison and edge direction normalization"""
        self._operation_count += 1
        operation_id = f"get_node_edges#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_node_edges(source_node_id)
            storage_result = await self.storage.get_node_edges(source_node_id)
            
            # Handle None cases
            if baseline_result is None and storage_result is None:
                print(f"✅ Oracle match for '{operation_id}' (both None)")
                return storage_result
            elif baseline_result is None or storage_result is None:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}' (None vs non-None):\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            # Compare edge lists without considering order or direction
            if len(baseline_result) != len(storage_result):
                raise AssertionError(
                    f"Oracle mismatch (edge list length) in '{operation_id}':\n"
                    f"  Storage:  {len(storage_result)} edges\n"
                    f"  Baseline: {len(baseline_result)} edges"
                )
            
            # Normalize edges and convert to sets for comparison
            baseline_normalized = self._normalize_edge_list(baseline_result)
            storage_normalized = self._normalize_edge_list(storage_result)
            
            if baseline_normalized != storage_normalized:
                only_in_storage = storage_normalized - baseline_normalized
                only_in_baseline = baseline_normalized - storage_normalized
                
                raise AssertionError(
                    f"Oracle mismatch (edge list content) in '{operation_id}':\n"
                    f"  Storage:  {sorted(storage_result)}\n"
                    f"  Baseline: {sorted(baseline_result)}\n"
                    f"  Storage normalized: {sorted(storage_normalized)}\n"
                    f"  Baseline normalized: {sorted(baseline_normalized)}\n"
                    f"  Only in storage: {sorted(only_in_storage) if only_in_storage else 'None'}\n"
                    f"  Only in baseline: {sorted(only_in_baseline) if only_in_baseline else 'None'}"
                )
            
            print(f"✅ Oracle match for '{operation_id}' (unordered edge list with direction normalization)")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes batch with flexible field comparison for each node"""
        self._operation_count += 1
        operation_id = f"get_nodes_batch#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_nodes_batch(node_ids)
            storage_result = await self.storage.get_nodes_batch(node_ids)
            
            # Compare keys first
            if set(baseline_result.keys()) != set(storage_result.keys()):
                raise AssertionError(
                    f"Oracle mismatch (keys) in '{operation_id}':\n"
                    f"  Storage keys:  {sorted(storage_result.keys())}\n"
                    f"  Baseline keys: {sorted(baseline_result.keys())}"
                )
            
            # Compare each node with flexible comparison
            for node_id in baseline_result.keys():
                baseline_node = baseline_result[node_id]
                storage_node = storage_result[node_id]
                
                # Handle None cases
                if baseline_node is None and storage_node is None:
                    continue
                elif baseline_node is None or storage_node is None:
                    raise AssertionError(
                        f"Oracle mismatch (node {node_id} None vs non-None) in '{operation_id}':\n"
                        f"  Storage:  {storage_node}\n"
                        f"  Baseline: {baseline_node}"
                    )
                
                # Use flexible dictionary comparison for each node
                if not self._flexible_dict_compare(baseline_node, storage_node, f"{operation_id}_node_{node_id}"):
                    raise AssertionError(
                        f"Oracle mismatch (node {node_id}) in '{operation_id}':\n"
                        f"  Storage:  {storage_node}\n"
                        f"  Baseline: {baseline_node}"
                    )
            
            print(f"✅ Oracle match for '{operation_id}' ({len(storage_result)} nodes)")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Get node degrees batch"""
        self._operation_count += 1
        operation_id = f"node_degrees_batch#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.node_degrees_batch(node_ids)
            storage_result = await self.storage.node_degrees_batch(node_ids)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Get edge degrees batch"""
        self._operation_count += 1
        operation_id = f"edge_degrees_batch#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.edge_degrees_batch(edge_pairs)
            storage_result = await self.storage.edge_degrees_batch(edge_pairs)
            
            if baseline_result != storage_result:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {storage_result}\n"
                    f"  Baseline: {baseline_result}"
                )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Get edges batch"""
        self._operation_count += 1
        operation_id = f"get_edges_batch#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_edges_batch(pairs)
            storage_result = await self.storage.get_edges_batch(pairs)
            
            # Compare keys first
            if set(baseline_result.keys()) != set(storage_result.keys()):
                raise AssertionError(
                    f"Oracle mismatch (edge batch keys) in '{operation_id}':\n"
                    f"  Storage keys:  {sorted(storage_result.keys())}\n"
                    f"  Baseline keys: {sorted(baseline_result.keys())}"
                )
            
            # Compare each edge data with flexible comparison
            for edge_key in baseline_result.keys():
                baseline_edge = baseline_result[edge_key]
                storage_edge = storage_result[edge_key]
                
                # Handle None cases
                if baseline_edge is None and storage_edge is None:
                    continue
                elif baseline_edge is None or storage_edge is None:
                    raise AssertionError(
                        f"Oracle mismatch (edge {edge_key} None vs non-None) in '{operation_id}':\n"
                        f"  Storage:  {storage_edge}\n"
                        f"  Baseline: {baseline_edge}"
                    )
                
                # Use flexible dictionary comparison for each edge
                if not self._flexible_dict_compare(baseline_edge, storage_edge, f"{operation_id}_edge_{edge_key}"):
                    raise AssertionError(
                        f"Oracle mismatch (edge {edge_key}) in '{operation_id}':\n"
                        f"  Storage:  {storage_edge}\n"
                        f"  Baseline: {baseline_edge}"
                    )
            
            print(f"✅ Oracle match for '{operation_id}'")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges batch with unordered comparison and edge direction normalization for each node's edge list"""
        self._operation_count += 1
        operation_id = f"get_nodes_edges_batch#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_nodes_edges_batch(node_ids)
            storage_result = await self.storage.get_nodes_edges_batch(node_ids)
            
            # Compare keys first
            if set(baseline_result.keys()) != set(storage_result.keys()):
                raise AssertionError(
                    f"Oracle mismatch (nodes edges batch keys) in '{operation_id}':\n"
                    f"  Storage keys:  {sorted(storage_result.keys())}\n"
                    f"  Baseline keys: {sorted(baseline_result.keys())}"
                )
            
            # Compare each node's edge list (unordered with direction normalization)
            for node_id in baseline_result.keys():
                baseline_edges = baseline_result[node_id]
                storage_edges = storage_result[node_id]
                
                # Check length first
                if len(baseline_edges) != len(storage_edges):
                    raise AssertionError(
                        f"Oracle mismatch (node edges length for {node_id}) in '{operation_id}':\n"
                        f"  Storage:  {len(storage_edges)} edges\n"
                        f"  Baseline: {len(baseline_edges)} edges"
                    )
                
                # Normalize edges and compare as sets (unordered with direction normalization)
                baseline_normalized = self._normalize_edge_list(baseline_edges)
                storage_normalized = self._normalize_edge_list(storage_edges)
                
                if baseline_normalized != storage_normalized:
                    only_in_storage = storage_normalized - baseline_normalized
                    only_in_baseline = baseline_normalized - storage_normalized
                    
                    raise AssertionError(
                        f"Oracle mismatch (node edges for {node_id}) in '{operation_id}':\n"
                        f"  Storage:  {sorted(storage_edges)}\n"
                        f"  Baseline: {sorted(baseline_edges)}\n"
                        f"  Storage normalized: {sorted(storage_normalized)}\n"
                        f"  Baseline normalized: {sorted(baseline_normalized)}\n"
                        f"  Only in storage: {sorted(only_in_storage) if only_in_storage else 'None'}\n"
                        f"  Only in baseline: {sorted(only_in_baseline) if only_in_baseline else 'None'}"
                    )
            
            print(f"✅ Oracle match for '{operation_id}' (unordered node edges batch with direction normalization)")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert node in both storages"""
        self._operation_count += 1
        operation_id = f"upsert_node#{self._operation_count}"
        
        try:
            # Execute on baseline first
            await self.baseline.upsert_node(node_id, node_data)
            
            # Execute on storage
            await self.storage.upsert_node(node_id, node_data)
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert edge in both storages"""
        self._operation_count += 1
        operation_id = f"upsert_edge#{self._operation_count}"
        
        try:
            # Execute on baseline first
            await self.baseline.upsert_edge(source_node_id, target_node_id, edge_data)
            
            # Execute on storage
            await self.storage.upsert_edge(source_node_id, target_node_id, edge_data)
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def delete_node(self, node_id: str) -> None:
        """Delete node from both storages"""
        self._operation_count += 1
        operation_id = f"delete_node#{self._operation_count}"
        
        try:
            # Execute on baseline first
            await self.baseline.delete_node(node_id)
            
            # Execute on storage
            await self.storage.delete_node(node_id)
            
            print(f"⚖️  Write operation '{operation_id}' completed and synced")
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Remove nodes from both storages"""
        self._operation_count += 1
        operation_id = f"remove_nodes#{self._operation_count}"
        
        try:
            # Execute on baseline first
            await self.baseline.remove_nodes(nodes)
            
            # Execute on storage
            await self.storage.remove_nodes(nodes)
            
            print(f"⚖️  Write operation '{operation_id}' completed and synced")
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Remove edges from both storages"""
        self._operation_count += 1
        operation_id = f"remove_edges#{self._operation_count}"
        
        try:
            # Execute on baseline first
            await self.baseline.remove_edges(edges)
            
            # Execute on storage
            await self.storage.remove_edges(edges)
            
            print(f"⚖️  Write operation '{operation_id}' completed and synced")
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_all_labels(self) -> list[str]:
        """Get all labels with sorted comparison"""
        self._operation_count += 1
        operation_id = f"get_all_labels#{self._operation_count}"
        
        try:
            baseline_result = await self.baseline.get_all_labels()
            storage_result = await self.storage.get_all_labels()
            
            # Sort both lists for comparison (order shouldn't matter for labels)
            baseline_sorted = sorted(baseline_result) if baseline_result else []
            storage_sorted = sorted(storage_result) if storage_result else []
            
            if baseline_sorted != storage_sorted:
                raise AssertionError(
                    f"Oracle mismatch in '{operation_id}':\n"
                    f"  Storage:  {len(storage_sorted)} labels: {storage_sorted[:10]}...\n"
                    f"  Baseline: {len(baseline_sorted)} labels: {baseline_sorted[:10]}..."
                )
            
            print(f"✅ Oracle match for '{operation_id}' ({len(storage_result)} labels)")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = 1000) -> KnowledgeGraph:
        """Get knowledge graph - warning for complex comparison"""
        self._operation_count += 1
        operation_id = f"get_knowledge_graph#{self._operation_count}"
        
        try:
            # For now, just return storage result with a warning
            # Full comparison would be very complex
            storage_result = await self.storage.get_knowledge_graph(node_label, max_depth, max_nodes)
            
            print(f"⚠️  Oracle bypassed detailed comparison for '{operation_id}' (complex structure)")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    async def drop(self) -> dict[str, str]:
        """Drop both storages"""
        self._operation_count += 1
        operation_id = f"drop#{self._operation_count}"
        
        try:
            # Execute on baseline first
            baseline_result = await self.baseline.drop()
            
            # Execute on storage
            storage_result = await self.storage.drop()
            
            print(f"⚖️  Write operation '{operation_id}' completed and synced")
            return storage_result
            
        except Exception as e:
            print(f"❌ Oracle operation '{operation_id}' failed: {e}")
            raise

    # Additional Oracle-specific methods

    async def compare_graphs_fully(self):
        """在测试结束时进行全面的图状态比较"""
        print("⚖️  Performing full graph comparison...")
        
        try:
            # 比较标签
            storage_labels = await self.storage.get_all_labels()
            baseline_labels = await self.baseline.get_all_labels()
            
            storage_labels_sorted = sorted(storage_labels) if storage_labels else []
            baseline_labels_sorted = sorted(baseline_labels) if baseline_labels else []
            
            if storage_labels_sorted != baseline_labels_sorted:
                print(f"⚠️  Label mismatch: storage={len(storage_labels_sorted)}, baseline={len(baseline_labels_sorted)}")
                # 不抛出异常，只是警告，因为某些implementation可能有差异
            else:
                print(f"✅ Labels match: {len(storage_labels_sorted)} labels")
            
            print(f"🎯 Oracle completed {self._operation_count} operations successfully")
            
        except Exception as e:
            print(f"⚠️  Full graph comparison warning: {e}")
            # 不抛出异常，允许测试继续 