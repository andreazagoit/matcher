import { GraphQLScalarType, Kind } from "graphql";
import { conversationResolvers } from "../models/conversations/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { tierResolvers } from "../models/tiers/resolver";
import { userResolvers } from "../models/users/resolver";

const JSONScalar = new GraphQLScalarType({
  name: "JSON",
  description: "Arbitrary JSON scalar",
  serialize: (value) => value,
  parseValue: (value) => value,
  parseLiteral: (ast) => {
    if (ast.kind === Kind.STRING) return JSON.parse(ast.value);
    if (ast.kind === Kind.INT) return parseInt(ast.value, 10);
    if (ast.kind === Kind.FLOAT) return parseFloat(ast.value);
    if (ast.kind === Kind.BOOLEAN) return ast.value;
    return null;
  },
});

export const resolvers = {

  JSON: JSONScalar,

  Query: {
    ...conversationResolvers.Query,
    ...matchResolvers.Query,
    ...postResolvers.Query,
    ...spaceResolvers.Query,
    ...userResolvers.Query,
  },

  Mutation: {
    ...conversationResolvers.Mutation,
    ...matchResolvers.Mutation,
    ...spaceResolvers.Mutation,
    ...userResolvers.Mutation,
  },

  // Relationship and specific type resolvers
  Conversation: conversationResolvers.Conversation,
  Member: { ...memberResolvers.Member, ...tierResolvers.Member },
  Message: conversationResolvers.Message,
  Post: postResolvers.Post,
  Space: {
    ...spaceResolvers.Space,
    ...memberResolvers.Space,
    ...postResolvers.Space,
    ...tierResolvers.Space
  },
};
