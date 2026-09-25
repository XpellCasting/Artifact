import unittest
from app.GAT_LSTM.graph import graph_from_f2_map


class GraphTests(unittest.TestCase):
    def test_orientation_and_top_k(self):
        rows = [{"source": "c", "target": "a", "accepted": True, "strength": .8,
                 "association_signed": .8},
                {"source": "c", "target": "b", "accepted": True, "strength": .9,
                 "association_signed": -.9},
                {"source": "a", "target": "c", "accepted": True, "strength": .1,
                 "association_signed": .1}]
        graph = graph_from_f2_map({"edges": rows}, ["c", "a", "b"], 1)
        self.assertEqual(graph["entity_ids"], ["a", "b", "c"])
        self.assertTrue(graph["adjacency"][2][1])  # c receives context from b (stronger)
        self.assertFalse(graph["adjacency"][2][0])
        self.assertTrue(graph["adjacency"][0][2])  # direction is asymmetric
        self.assertFalse(graph["adjacency"][0][1])
        self.assertTrue(all(graph["adjacency"][i][i] for i in range(3)))


if __name__ == "__main__":
    unittest.main()
