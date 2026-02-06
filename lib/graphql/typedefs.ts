import { userTypeDefs } from "../models/users/typedefs";
import { spaceTypeDefs } from "../models/spaces/typedefs";
import { memberTypeDefs } from "../models/members/typedefs";
import { postTypeDefs } from "../models/posts/typedefs";

import { tierTypeDefs } from "../models/tiers/typedefs";

// Base types - definizioni vuote che vengono estese dai modelli
const baseTypeDefs = `#graphql
  """
  JSON scalar per dati complessi (traits, etc)
  """
  scalar JSON

  type Query
  type Mutation
`;

export const typeDefs = [baseTypeDefs, userTypeDefs, spaceTypeDefs, memberTypeDefs, postTypeDefs, tierTypeDefs];
