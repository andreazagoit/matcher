export const spaceTypeDefs = `#graphql
  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    image: String
    categories: [String!]!
    visibility: String!
    joinPolicy: String!
    createdAt: DateTime!
    isActive: Boolean
    membersCount: Int
    type: String
    stripeAccountEnabled: Boolean!
  }

  input CreateSpaceInput {
    name: String!
    slug: String
    description: String
    visibility: String
    joinPolicy: String
    categories: [String!]
  }

  input UpdateSpaceInput {
    name: String
    description: String
    visibility: String
    joinPolicy: String
    image: String
    categories: [String!]
  }

  extend type Query {
    space(id: ID, slug: String): Space
    spaces: [Space!]!
    mySpaces: [Space!]!

    """
    Search public spaces by categories.
    matchAll=true requires ALL categories, false requires at least one.
    """
    spacesByCategories(categories: [String!]!, matchAll: Boolean): [Space!]!

    """
    Get recommended spaces based on behavioral similarity and category overlap.
    Excludes spaces the user is already a member of.
    """
    recommendedSpaces(limit: Int): [Space!]!
  }

  extend type Mutation {
    createSpace(input: CreateSpaceInput!): Space!
    updateSpace(id: ID!, input: UpdateSpaceInput!): Space!
    deleteSpace(id: ID!): Boolean!
  }
`;
