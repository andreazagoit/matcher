import { conversationResolvers } from "../models/conversations/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { tierResolvers } from "../models/tiers/resolver";
import { userResolvers } from "../models/users/resolver";

export const resolvers = {

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
