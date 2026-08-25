import unittest

from ontology.concept_ontology import ConceptOntology, load_ontology


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
        self.assertNotIn("facts", ontology.bim_query_knowledge)
        rendered = str(ontology.bim_query_knowledge)
        self.assertNotIn("There are 70", rendered)
        self.assertNotIn("100 distinct values", rendered)
        self.assertNotIn("about 45 percent", rendered)

    def test_unrelated_project_does_not_receive_current_project_knowledge(self):
        ontology = load_ontology(
            client_id="unrelated-client",
            project_id="unrelated-project",
            reload=True,
        )
        self.assertEqual(ontology.bim_query_knowledge, {})

    def test_electrical_project_loads_switch_semantics_without_answer_values(self):
        ontology = load_ontology(
            client_id="2efb2eae-6933-4075-abef-9aaf43939187",
            project_id="d1055960-9a3c-11f1-abb9-ef657ab9d228",
            reload=True,
        )

        mapping = ontology.bim_query_knowledge["lighting_switches"]
        self.assertEqual(mapping["identity_property"], "GlobalID")
        self.assertIn("M_Lighting Switches_Y23", mapping["exact_family_values"])
        self.assertNotIn("count", mapping)
        self.assertNotIn("value", mapping)

    def test_direct_answers_are_rejected_from_future_knowledge_overlays(self):
        with self.assertRaisesRegex(ValueError, "forbidden direct-answer"):
            ConceptOntology({"bim_query_knowledge": {"facts": {"answer": {"value": 70}}}})
        with self.assertRaisesRegex(ValueError, "stores a direct answer"):
            ConceptOntology({
                "bim_query_knowledge": {
                    "program": {"statement": "There are 70 apartments.", "value": 70}
                }
            })

    def test_global_mep_vocabulary_expands_hebrew_questions_to_model_terms(self):
        ontology = load_ontology(reload=True)

        self.assertIn("Lighting Devices", ontology.retrieval_terms_for("כמה מפסקים בפרויקט?"))
        self.assertIn("Cable Tray", ontology.retrieval_terms_for("אילו מגשי כבלים קיימים?"))
        self.assertIn("Conduit", ontology.retrieval_terms_for("כמה צינורות חשמל יש?"))


if __name__ == "__main__":
    unittest.main()
