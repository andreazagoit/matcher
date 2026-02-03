import { userTypeDefs } from "../models/users/typedefs";

// Base types - definizioni vuote che vengono estese dai modelli
const baseTypeDefs = `#graphql
  type Query
  type Mutation
`;

export const typeDefs = [baseTypeDefs, userTypeDefs];
