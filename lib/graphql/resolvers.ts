import { GraphQLScalarType, Kind } from "graphql";
import { userResolvers } from "../models/users/resolver";

// JSON Scalar per dati complessi (traits, etc)
const JSONScalar = new GraphQLScalarType({
  name: "JSON",
  description: "JSON custom scalar type",
  serialize(value) {
    return value;
  },
  parseValue(value) {
    return value;
  },
  parseLiteral(ast) {
    if (ast.kind === Kind.STRING) {
      return JSON.parse(ast.value);
    }
    return null;
  },
});

export const resolvers = {
  JSON: JSONScalar,
  
  Query: {
    ...userResolvers.Query,
  },
  Mutation: {
    ...userResolvers.Mutation,
  },
  
  // Field resolvers
  User: userResolvers.User,
};

