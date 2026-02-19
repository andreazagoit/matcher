export const spaceTypeDefs = `#graphql
  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    image: String
    tags: [String!]!
    visibility: String!
    joinPolicy: String!
    createdAt: String!
    isActive: Boolean
    membersCount: Int
    type: String
  }

  input CreateSpaceInput {
    name: String!
    slug: String
    description: String
    visibility: String
    joinPolicy: String
    tags: [String!]
  }

  input UpdateSpaceInput {
    name: String
    description: String
    visibility: String
    joinPolicy: String
    image: String
    tags: [String!]
  }

  extend type Query {
    space(id: ID, slug: String): Space
    spaces: [Space!]!
    mySpaces: [Space!]!

    """
    Search public spaces by tags.
    matchAll=true requires ALL tags, false requires at least one.
    """
    spacesByTags(tags: [String!]!, matchAll: Boolean): [Space!]!

    """
    Get recommended spaces based on behavioral similarity and tag overlap.
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
