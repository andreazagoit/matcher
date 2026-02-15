import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const USER_FRAGMENT = gql`
  fragment UserFields on User {
    id
    givenName
    familyName
    email
    birthdate
    gender
    image
    createdAt
    updatedAt
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_ME = gql`
  ${USER_FRAGMENT}
  query GetMe {
    me {
      ...UserFields
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const UPDATE_USER = gql`
  ${USER_FRAGMENT}
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      ...UserFields
    }
  }
`;

