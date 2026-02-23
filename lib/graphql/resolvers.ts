import { conversationResolvers } from "../models/conversations/resolver";
import { notificationResolvers } from "../models/notifications/resolver";
import { eventResolvers } from "../models/events/resolver";
import { interestResolvers } from "../models/interests/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { userItemResolvers } from "../models/profileitems/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { tagResolvers } from "../models/tags/resolver";
import { tierResolvers } from "../models/tiers/resolver";
import { userResolvers } from "../models/users/resolver";



export const resolvers = {


  Query: {
    ...conversationResolvers.Query,
    ...eventResolvers.Query,
    ...interestResolvers.Query,
    ...matchResolvers.Query,
    ...postResolvers.Query,
    ...userItemResolvers.Query,
    ...spaceResolvers.Query,
    ...tagResolvers.Query,
    ...userResolvers.Query,
    ...notificationResolvers.Query,
  },

  Mutation: {
    ...conversationResolvers.Mutation,
    ...eventResolvers.Mutation,
    ...interestResolvers.Mutation,
    ...memberResolvers.Mutation,
    ...postResolvers.Mutation,
    ...userItemResolvers.Mutation,
    ...spaceResolvers.Mutation,
    ...userResolvers.Mutation,
    ...notificationResolvers.Mutation,
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
