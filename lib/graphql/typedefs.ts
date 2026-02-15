import { matchTypeDefs } from "../models/matches/typedefs";
import { memberTypeDefs } from "../models/members/typedefs";
import { postTypeDefs } from "../models/posts/typedefs";
import { spaceTypeDefs } from "../models/spaces/typedefs";
import { tierTypeDefs } from "../models/tiers/typedefs";
import { userTypeDefs } from "../models/users/typedefs";
import { conversationTypeDefs } from "../models/conversations/typedefs";

// Base types - empty definitions that are extended by model-specific typedefs
const baseTypeDefs = `#graphql
  scalar JSON

  type Query
  type Mutation
`;

export const typeDefs = [
  baseTypeDefs,
  conversationTypeDefs,
  matchTypeDefs,
  memberTypeDefs,
  postTypeDefs,
  spaceTypeDefs,
  tierTypeDefs,
  userTypeDefs,
];
