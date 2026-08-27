const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

const dataDir = path.join(__dirname, "test-project-data");
const treePath = path.join(
  dataDir,
  "3b230392-60e7-f2b2-a0d0-6fcd9ee45815-tree.json",
);
const propertiesPath = path.join(
  dataDir,
  "3b230392-60e7-f2b2-a0d0-6fcd9ee45815-properties.json",
);
const sqlitePath = path.join(
  dataDir,
  "07bd11f0-9a3d-11f1-abb9-ef657ab9d228.ifc",
);

const tree = JSON.parse(fs.readFileSync(treePath, "utf8"));
const properties = JSON.parse(fs.readFileSync(propertiesPath, "utf8"));
const root = tree.data.objects[0];

function children(node) {
  return Array.isArray(node.objects) ? node.objects : [];
}

function leaves(node) {
  const descendants = children(node);
  if (descendants.length === 0) return [node];
  return descendants.flatMap(leaves);
}

function flattenStrings(value, fieldPath = "", result = []) {
  if (typeof value === "string") {
    result.push({ fieldPath, value });
  } else if (Array.isArray(value)) {
    value.forEach((item, index) =>
      flattenStrings(item, `${fieldPath}[${index}]`, result),
    );
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      const nextPath = fieldPath ? `${fieldPath}.${key}` : key;
      // Search property names as well as their values.
      result.push({ fieldPath: nextPath, value: key });
      flattenStrings(item, nextPath, result);
    }
  }
  return result;
}

function matchingRecords(pattern) {
  const matches = [];
  for (const [objectId, record] of Object.entries(properties)) {
    const fields = flattenStrings(record);
    const matchingFields = fields.filter(({ value }) => pattern.test(value));
    if (matchingFields.length > 0) {
      matches.push({
        objectId,
        name: record.name,
        externalId: record.externalId,
        matches: matchingFields,
      });
    }
  }
  return matches;
}

const switchWords = /switch|breaker|disconnect(?:or)?|isolat(?:or|ing)|contactor|מפס/iu;
const switchCategory = /electrical fixtures|lighting devices|switching devices|ifcswitchingdevice/iu;
const lexicalMatches = matchingRecords(switchWords);
const categoryMatches = matchingRecords(switchCategory);

const branchSummary = children(root).map((branch) => {
  const branchLeaves = leaves(branch);
  return {
    objectId: branch.objectid,
    name: branch.name,
    leafCount: branchLeaves.length,
    uniqueLeafObjectIds: new Set(branchLeaves.map((leaf) => leaf.objectid)).size,
    leavesWithProperties: branchLeaves.filter(
      (leaf) => properties[String(leaf.objectid)] !== undefined,
    ).length,
  };
});

const allLeaves = leaves(root);
const physicalCategoryNames = new Set([
  "Electrical Fixtures",
  "Lighting Devices",
  "Switching Devices",
]);
const switchBranches = children(root).filter((branch) =>
  physicalCategoryNames.has(branch.name),
);
const switchLeaves = switchBranches.flatMap(leaves);

const database = new DatabaseSync(sqlitePath, { readOnly: true });
const sqliteSchema = database
  .prepare(
    "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name",
  )
  .all();
const sqliteTableCounts = Object.fromEntries(
  sqliteSchema.map(({ name }) => [
    name,
    database.prepare(`SELECT count(*) AS count FROM "${name}"`).get().count,
  ]),
);
const sqliteSwitchValues = database
  .prepare(
    `SELECT v.id AS valueId,
            CAST(v.value AS TEXT) AS valueText,
            count(e.id) AS referenceCount,
            count(DISTINCT e.entity_id) AS entityCount
       FROM _objects_val AS v
       LEFT JOIN _objects_eav AS e ON e.value_id = v.id
      WHERE lower(CAST(v.value AS TEXT)) LIKE '%switch%'
         OR CAST(v.value AS TEXT) LIKE '%מפס%'
      GROUP BY v.id, v.value
      ORDER BY entityCount DESC, valueId`,
  )
  .all()
  .map((row) => ({
    ...row,
    valueText:
      typeof row.valueText === "string"
        ? row.valueText.replace(/[\u0000-\u001f]/gu, " ").trim()
        : row.valueText,
  }));
const sqliteCandidateEntities = database
  .prepare(
    `SELECT DISTINCT e.entity_id AS entityId,
            a.name AS attributeName,
            a.category AS attributeCategory,
            CAST(v.value AS TEXT) AS matchingValue
       FROM _objects_eav AS e
       JOIN _objects_attr AS a ON a.id = e.attribute_id
       JOIN _objects_val AS v ON v.id = e.value_id
      WHERE lower(CAST(v.value AS TEXT)) LIKE '%switch%'
         OR CAST(v.value AS TEXT) LIKE '%מפס%'
      ORDER BY e.entity_id, a.category, a.name`,
  )
  .all()
  .map((row) => ({
    ...row,
    matchingValue:
      typeof row.matchingValue === "string"
        ? row.matchingValue.replace(/[\u0000-\u001f]/gu, " ").trim()
        : row.matchingValue,
    presentInExportedProperties:
      properties[String(row.entityId)] !== undefined,
  }));
const candidateEntityIds = [
  ...new Set(sqliteCandidateEntities.map(({ entityId }) => entityId)),
];
const sqliteCandidateDetails = candidateEntityIds.map((entityId) => {
  const identity = database
    .prepare(
      `SELECT id,
              CAST(external_id AS TEXT) AS externalId,
              CAST(viewable_id AS TEXT) AS viewableId,
              length(external_id) AS externalIdBytes,
              length(viewable_id) AS viewableIdBytes
         FROM _objects_id
        WHERE id = ?`,
    )
    .get(entityId);
  const eavCount = database
    .prepare("SELECT count(*) AS count FROM _objects_eav WHERE entity_id = ?")
    .get(entityId).count;
  const attributes = database
    .prepare(
      `SELECT a.category, a.name,
              CAST(v.value AS TEXT) AS value
         FROM _objects_eav AS e
         JOIN _objects_attr AS a ON a.id = e.attribute_id
         JOIN _objects_val AS v ON v.id = e.value_id
        WHERE e.entity_id = ?
        ORDER BY a.category, a.name
        LIMIT 200`,
    )
    .all(entityId)
    .map((attribute) => ({
      ...attribute,
      value:
        typeof attribute.value === "string"
          ? attribute.value.replace(/[\u0000-\u001f]/gu, " ").trim()
        : attribute.value,
    }));
  const parentId = Number(
    attributes.find(
      ({ category, name }) => category === "__parent__" && name === "parent",
    )?.value,
  );
  const parent = Number.isFinite(parentId)
    ? database
        .prepare(
          `SELECT e.entity_id AS entityId, CAST(v.value AS TEXT) AS name
             FROM _objects_eav AS e
             JOIN _objects_attr AS a ON a.id = e.attribute_id
             JOIN _objects_val AS v ON v.id = e.value_id
            WHERE e.entity_id = ?
              AND a.category = '__name__'
              AND a.name = 'name'
            LIMIT 1`,
        )
        .get(parentId)
    : null;
  return {
    entityId,
    identity,
    eavCount,
    attributes,
    parent: parent || null,
    presentInTree: allLeaves.some((leaf) => leaf.objectid === entityId),
    presentInExportedProperties: properties[String(entityId)] !== undefined,
  };
});
database.close();

const report = {
  sources: {
    sqliteModel: path.relative(__dirname, sqlitePath),
    tree: path.relative(__dirname, treePath),
    properties: path.relative(__dirname, propertiesPath),
  },
  propertyRecordCount: Object.keys(properties).length,
  sqliteSchema,
  sqliteTableCounts,
  sqliteSwitchValues,
  sqliteCandidateEntities,
  sqliteCandidateDetails,
  rootName: root.name,
  branchSummary,
  treeIntegrity: {
    totalLeaves: allLeaves.length,
    uniqueLeafObjectIds: new Set(allLeaves.map((leaf) => leaf.objectid)).size,
    leavesWithProperties: allLeaves.filter(
      (leaf) => properties[String(leaf.objectid)] !== undefined,
    ).length,
  },
  switchEvidence: {
    lexicalPattern: switchWords.source,
    lexicalMatchCount: lexicalMatches.length,
    lexicalMatches,
    categoryPattern: switchCategory.source,
    categoryMatchCount: categoryMatches.length,
    categoryMatches,
    matchingTopLevelCategories: switchBranches.map((branch) => branch.name),
    physicalSwitchLeafCount: switchLeaves.length,
    uniquePhysicalSwitchObjectIds: new Set(
      switchLeaves.map((leaf) => leaf.objectid),
    ).size,
  },
};

console.log(JSON.stringify(report, null, 2));
