export const spaceTypeDefs = `#graphql
  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    logoUrl: String
    membersCount: Int
    isPublic: Boolean
    requiresApproval: Boolean
    createdAt: String!
    ownerId: ID!
    owner: User!
    clientId: String
    isActive: Boolean
  }

  input CreateSpaceInput {
    name: String!
    slug: String
    description: String
    isPublic: Boolean
  }

  input UpdateSpaceInput {
    name: String
    description: String
    isPublic: Boolean
    requiresApproval: Boolean
    logoUrl: String
  }

  extend type Query {
    space(id: ID, slug: String): Space
    spaces: [Space!]!
    mySpaces: [Space!]!
  }

  extend type Mutation {
    createSpace(input: CreateSpaceInput!): Space!
    updateSpace(id: ID!, input: UpdateSpaceInput!): Space!
    deleteSpace(id: ID!): Boolean!
  }
`;
