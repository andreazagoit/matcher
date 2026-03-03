export const tierTypeDefs = `#graphql
  enum TierInterval {
    month
    year
    one_time
  }

  type MembershipTier {
    id: ID!
    name: String!
    description: String
    price: Int!
    currency: String!
    interval: TierInterval!
    isActive: Boolean!
    spaceId: ID!
  }

  input CreateTierInput {
    name: String!
    description: String
    price: Int!
    interval: TierInterval!
  }

  input UpdateTierInput {
    name: String
    description: String
    price: Int
    interval: TierInterval
    isActive: Boolean
  }

  extend type Space {
    tiers: [MembershipTier!]!
  }

  extend type Member {
    tier: MembershipTier
  }

  extend type Mutation {
    createTier(spaceId: ID!, input: CreateTierInput!): MembershipTier!
    updateTier(id: ID!, input: UpdateTierInput!): MembershipTier!
    archiveTier(id: ID!): Boolean!
  }
`;
