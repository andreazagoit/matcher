import { userTypeDefs } from "../models/users/typedefs";

// Base types - definizioni vuote che vengono estese dai modelli
const baseTypeDefs = `#graphql
  """
  JSON scalar per dati complessi (traits, etc)
  """
  scalar JSON

  type Query
  type Mutation
`;

export const typeDefs = [baseTypeDefs, userTypeDefs];
