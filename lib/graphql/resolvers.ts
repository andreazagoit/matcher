import { GraphQLScalarType, Kind } from "graphql";
import { conversationResolvers } from "../models/conversations/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { tierResolvers } from "../models/tiers/resolver";
import { userResolvers } from "../models/users/resolver";

// JSON Scalar for complex data (traits, metadata, etc.)
const JSONScalar = new GraphQLScalarType({
  name: "JSON",
  description: "JSON custom scalar type",
  serialize(value) { return value; },
  parseValue(value) { return value; },
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
    ...conversationResolvers.Query,
    ...matchResolvers.Query,
    ...postResolvers.Query,
    ...spaceResolvers.Query,
    ...userResolvers.Query,
  },

  Mutation: {
    ...conversationResolvers.Mutation,
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
  User: userResolvers.User,
};
