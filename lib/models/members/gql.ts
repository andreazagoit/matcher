import { gql } from "graphql-tag";

// ============================================
// MUTATIONS
// ============================================

export const UPDATE_MEMBER_ROLE = gql`
  mutation UpdateMemberRole($spaceId: ID!, $userId: ID!, $role: String!) {
    updateMemberRole(spaceId: $spaceId, userId: $userId, role: $role) {
      id
      role
    }
  }
`;

export const REMOVE_MEMBER = gql`
  mutation RemoveMember($spaceId: ID!, $userId: ID!) {
    removeMember(spaceId: $spaceId, userId: $userId)
  }
`;
