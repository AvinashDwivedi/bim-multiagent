import unittest

from ontology.concept_ontology import load_ontology


class ScopedKnowledgeTests(unittest.TestCase):
    def test_current_project_loads_bim_query_knowledge(self):
        ontology = load_ontology(
            client_id="653fbe80-e4c5-11ed-95e8-fdb8a484b2c4",
            project_id="858ef0f0-454a-11f1-8957-1fe1b101e373",
            reload=True,
        )
        self.assertEqual(
            ontology.bim_query_knowledge["area_plan"]["floor_plate_basis"], "BVO"
        )
        self.assertEqual(
            ontology.bim_query_knowledge["tower_facade"]["wall_name_value"], "Klimaatgevel"
        )

    def test_unrelated_project_does_not_receive_current_project_knowledge(self):
        ontology = load_ontology(
            client_id="unrelated-client",
            project_id="unrelated-project",
            reload=True,
        )
        self.assertEqual(ontology.bim_query_knowledge, {})


if __name__ == "__main__":
    unittest.main()
