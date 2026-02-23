import { conversationTypeDefs } from "../models/conversations/typedefs";
import { notificationTypeDefs } from "../models/notifications/typedefs";
import { eventTypeDefs } from "../models/events/typedefs";
import { interestTypeDefs } from "../models/interests/typedefs";
import { matchTypeDefs } from "../models/matches/typedefs";
import { memberTypeDefs } from "../models/members/typedefs";
import { postTypeDefs } from "../models/posts/typedefs";
import { userItemTypeDefs } from "../models/profileitems/typedefs";
import { spaceTypeDefs } from "../models/spaces/typedefs";
import { tagTypeDefs } from "../models/tags/typedefs";
import { tierTypeDefs } from "../models/tiers/typedefs";
import { userTypeDefs } from "../models/users/typedefs";

// Base types - empty definitions that are extended by model-specific typedefs
const baseTypeDefs = `#graphql
  scalar DateTime
  scalar JSON

  type Query
  type Mutation
`;

export const typeDefs = [
  baseTypeDefs,
  conversationTypeDefs,
  eventTypeDefs,
  interestTypeDefs,
  matchTypeDefs,
  memberTypeDefs,
  postTypeDefs,
  userItemTypeDefs,
  spaceTypeDefs,
  tagTypeDefs,
  tierTypeDefs,
  userTypeDefs,
  notificationTypeDefs,
];
