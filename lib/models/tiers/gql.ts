import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const TIER_FRAGMENT = gql`
  fragment TierFields on MembershipTier {
    id
    name
    description
    price
    currency
    interval
    isActive
    spaceId
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_SPACE_TIERS = gql`
  ${TIER_FRAGMENT}
  query GetSpaceTiers($spaceId: ID!) {
    space(id: $spaceId) {
      id
      tiers {
        ...TierFields
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_TIER = gql`
  mutation CreateTier($spaceId: ID!, $input: CreateTierInput!) {
    createTier(spaceId: $spaceId, input: $input) {
      id
    }
  }
`;

export const ARCHIVE_TIER = gql`
  mutation ArchiveTier($id: ID!) {
    archiveTier(id: $id)
  }
`;
