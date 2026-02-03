import { userTypeDefs } from "./user";

const baseTypeDefs = `#graphql
  type Query {
    user(id: ID!): User
    users: [User!]!
    me: User
    findMatches(userId: ID!, limit: Int = 10): [User!]!
  }

  type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
  }
`;

export const typeDefs = [baseTypeDefs, userTypeDefs];
