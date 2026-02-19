import { conversationTypeDefs } from "../models/conversations/typedefs";
import { eventTypeDefs } from "../models/events/typedefs";
import { interestTypeDefs } from "../models/interests/typedefs";
import { matchTypeDefs } from "../models/matches/typedefs";
import { memberTypeDefs } from "../models/members/typedefs";
import { postTypeDefs } from "../models/posts/typedefs";
import { profileTypeDefs } from "../models/profiles/typedefs";
import { spaceTypeDefs } from "../models/spaces/typedefs";
import { tagTypeDefs } from "../models/tags/typedefs";
import { tierTypeDefs } from "../models/tiers/typedefs";
import { userTypeDefs } from "../models/users/typedefs";

// Base types - empty definitions that are extended by model-specific typedefs
const baseTypeDefs = `#graphql
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
  profileTypeDefs,
  spaceTypeDefs,
  tagTypeDefs,
  tierTypeDefs,
  userTypeDefs,
];
