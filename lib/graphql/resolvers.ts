import { GraphQLScalarType, Kind } from "graphql";
import { conversationResolvers } from "../models/conversations/resolver";
import { eventResolvers } from "../models/events/resolver";
import { interestResolvers } from "../models/interests/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { profileResolvers } from "../models/profiles/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { tagResolvers } from "../models/tags/resolver";
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
    ...eventResolvers.Query,
    ...interestResolvers.Query,
    ...matchResolvers.Query,
    ...postResolvers.Query,
    ...profileResolvers.Query,
    ...spaceResolvers.Query,
    ...tagResolvers.Query,
    ...userResolvers.Query,
  },

  Mutation: {
    ...conversationResolvers.Mutation,
    ...eventResolvers.Mutation,
    ...interestResolvers.Mutation,
    ...spaceResolvers.Mutation,
    ...userResolvers.Mutation,
  },

  // Relationship and specific type resolvers
  Conversation: conversationResolvers.Conversation,
  Event: eventResolvers.Event,
  EventAttendee: eventResolvers.EventAttendee,
  Member: { ...memberResolvers.Member, ...tierResolvers.Member },
  Message: conversationResolvers.Message,
  Post: postResolvers.Post,
  Space: {
    ...spaceResolvers.Space,
    ...memberResolvers.Space,
    ...postResolvers.Space,
    ...tierResolvers.Space
  },
  User: {
    ...userResolvers.User,
  },
};
