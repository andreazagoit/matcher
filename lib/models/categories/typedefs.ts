export const categoryTypeDefs = `#graphql
  type Category {
    id: String!
    name: String!
  }

  extend type Query {
    """All available interest categories."""
    categories: [Category!]!
    """Single category by id."""
    category(id: String!): Category
  }

  extend type Mutation {
    """Create a new category with ML embeddings (64d + 256d)."""
    createCategory(name: String!): Category!
  }
`;
