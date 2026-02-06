import { GraphQLScalarType, Kind } from "graphql";
import { userResolvers } from "../models/users/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { tierResolvers } from "../models/tiers/resolver";

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
    ...spaceResolvers.Query,
    ...postResolvers.Query,
  },
  Mutation: {
    ...userResolvers.Mutation,
    ...spaceResolvers.Mutation,
    ...memberResolvers.Mutation,
    ...postResolvers.Mutation,
    ...tierResolvers.Mutation,
  },

  // Field resolvers
  User: userResolvers.User,
  Space: {
    ...spaceResolvers.Space,
    ...memberResolvers.Space,
    ...postResolvers.Space,
    ...tierResolvers.Space
  },
  Member: { ...memberResolvers.Member, ...tierResolvers.Member },
  Post: postResolvers.Post,
};
