export const categoryTypeDefs = `#graphql
  type Category {
    id: ID!
    """AI-recommended events for this category."""
    recommendedEvents(limit: Int, offset: Int): [Event!]!
    """AI-recommended spaces for this category."""
    recommendedSpaces(limit: Int, offset: Int): [Space!]!
    """Categories with similar embeddings."""
    recommendedCategories(limit: Int): [String!]!
  }

  extend type Query {
    """All available interest categories (unique, sorted)."""
    categories: [Category!]!
    """Single category by slug id, null if not found."""
    category(id: ID!): Category
  }

  extend type Mutation {
    """Create a new category with ML embeddings (64d + 256d). Returns the category id."""
    createCategory(name: String!): String!
  }
`;
