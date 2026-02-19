/**
 * GraphQL Schema for Users
 */

export const userTypeDefs = `#graphql
  """
  User — demographic, auth, and location data
  """
  type User {
    id: ID!
    username: String
    givenName: String!
    familyName: String!
    email: String!
    birthdate: String!
    gender: Gender
    image: String
    location: Location
    locationUpdatedAt: DateTime
    createdAt: DateTime!
    updatedAt: DateTime!
    interests: [UserInterest!]!
  }

  type Location {
    lat: Float!
    lon: Float!
  }

  input CreateUserInput {
    givenName: String!
    familyName: String!
    email: String!
    birthdate: String!
    gender: Gender
  }

  input UpdateUserInput {
    username: String
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
    user(username: String!): User
    users: [User!]!
    me: User
    checkUsername(username: String!): Boolean!
  }

  extend type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
    updateLocation(lat: Float!, lon: Float!): User!
  }
`;
