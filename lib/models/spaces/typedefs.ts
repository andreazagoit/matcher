export const spaceTypeDefs = `#graphql
  type Space {
    id: ID!
    name: String!
    slug: String!
    description: String
    image: String
    visibility: String!
    joinPolicy: String!
    createdAt: String!
    clientId: String
    isActive: Boolean
    membersCount: Int
  }

  input CreateSpaceInput {
    name: String!
    slug: String
    description: String
    visibility: String
    joinPolicy: String
  }

  input UpdateSpaceInput {
    name: String
    description: String
    visibility: String
    joinPolicy: String
    image: String
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
