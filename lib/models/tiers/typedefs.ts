export const tierTypeDefs = `#graphql
  type MembershipTier {
    id: ID!
    name: String!
    description: String
    price: Int!
    currency: String!
    interval: String!
    isActive: Boolean!
    spaceId: ID!
  }

  input CreateTierInput {
    name: String!
    description: String
    price: Int!
    interval: String!
  }

  input UpdateTierInput {
    name: String
    description: String
    price: Int
    interval: String
    isActive: Boolean
  }

  extend type Space {
    tiers: [MembershipTier!]
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
