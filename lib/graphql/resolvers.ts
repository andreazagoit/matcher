import { connectionResolvers } from "../models/connections/resolver";
import { notificationResolvers } from "../models/notifications/resolver";
import { eventResolvers } from "../models/events/resolver";
import { matchResolvers } from "../models/matches/resolver";
import { memberResolvers } from "../models/members/resolver";
import { postResolvers } from "../models/posts/resolver";
import { userItemResolvers } from "../models/useritems/resolver";
import { spaceResolvers } from "../models/spaces/resolver";
import { categoryResolvers } from "../models/categories/resolver";
import { tierResolvers } from "../models/tiers/resolver";
import { userResolvers } from "../models/users/resolver";



export const resolvers = {


  Query: {
    ...eventResolvers.Query,
    ...spaceResolvers.Query,
    ...categoryResolvers.Query,
    ...userResolvers.Query,
  },

  Mutation: {
    ...categoryResolvers.Mutation,
    ...connectionResolvers.Mutation,
    ...eventResolvers.Mutation,
    ...memberResolvers.Mutation,
    ...postResolvers.Mutation,
    ...userItemResolvers.Mutation,
    ...spaceResolvers.Mutation,
    ...userResolvers.Mutation,
    ...notificationResolvers.Mutation,
  },

  // Relationship and specific type resolvers
  Category: categoryResolvers.Category,
  Connection: connectionResolvers.Connection,
  Event: eventResolvers.Event,
  EventAttendee: eventResolvers.EventAttendee,
  Member: { ...memberResolvers.Member, ...tierResolvers.Member },
  Message: connectionResolvers.Message,
  Post: postResolvers.Post,
  Space: {
    ...spaceResolvers.Space,
    ...memberResolvers.Space,
    ...postResolvers.Space,
    ...tierResolvers.Space
  },
  User: {
    ...userResolvers.User,
    ...notificationResolvers.User,
    ...userItemResolvers.User,
    ...matchResolvers.User,
    ...postResolvers.User,
    ...connectionResolvers.User,
  },
};
