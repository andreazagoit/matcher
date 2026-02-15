/**
 * GraphQL Schema for Users
 *
 * User: local demographic + auth data (stored in matcher DB).
 * Profile/assessment/matching data comes from Identity Matcher (see matches module).
 */

export const userTypeDefs = `#graphql
  """
  Local user — demographic + auth data
  """
  type User {
    id: ID!
    givenName: String!
    familyName: String!
    email: String!
    birthdate: String!
    gender: Gender
    image: String
    createdAt: String!
    updatedAt: String!
  }

  input CreateUserInput {
    givenName: String!
    familyName: String!
    email: String!
    birthdate: String!
    gender: Gender
  }

  input UpdateUserInput {
    givenName: String
    familyName: String
    email: String
    birthdate: String
    gender: Gender
  }

  enum Gender {
    man
    woman
    non_binary
  }

  extend type Query {
    user(id: ID!): User
    users: [User!]!
    me: User
  }

  extend type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
  }
`;
