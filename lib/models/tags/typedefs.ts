/**
 * GraphQL Schema for Tags
 */

export const tagTypeDefs = `#graphql
  type TagCategoryEntry {
    category: String!
    tags: [String!]!
  }

  extend type Query {
    """
    Get tag categories (shared vocabulary for profiles, events, spaces).
    """
    tagCategories: [TagCategoryEntry!]!

    """
    Get all valid tags as a flat list.
    """
    allTags: [String!]!
  }
`;
