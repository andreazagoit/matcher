import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const USER_FRAGMENT = gql`
  fragment UserFields on User {
    id
    firstName
    lastName
    email
    birthDate
    gender
    createdAt
    updatedAt
  }
`;

export const PROFILE_FRAGMENT = gql`
  fragment ProfileFields on Profile {
    id
    userId

    psychologicalDescription
    valuesDescription
    interestsDescription
    behavioralDescription
    createdAt
    updatedAt
    embeddingsComputedAt
  }
`;

export const USER_WITH_PROFILE_FRAGMENT = gql`
  ${USER_FRAGMENT}
  ${PROFILE_FRAGMENT}
  fragment UserWithProfile on User {
    ...UserFields
    profile {
      ...ProfileFields
    }
  }
`;

export const USER_MATCH_FRAGMENT = gql`
  ${USER_FRAGMENT}
  fragment UserMatchFields on UserMatch {
    user {
      ...UserFields
    }
    similarity
    breakdown {
      psychological
      values
      interests
      behavioral
    }
  }
`;

// ============================================
// QUERIES
// ============================================

export const GET_USER = gql`
  ${USER_FRAGMENT}
  query GetUser($id: ID!) {
    user(id: $id) {
      ...UserFields
    }
  }
`;

export const GET_USER_WITH_PROFILE = gql`
  ${USER_WITH_PROFILE_FRAGMENT}
  query GetUserWithProfile($id: ID!) {
    user(id: $id) {
      ...UserWithProfile
    }
  }
`;

export const GET_ALL_USERS = gql`
  ${USER_FRAGMENT}
  query GetAllUsers {
    users {
      ...UserFields
    }
  }
`;

export const GET_ME = gql`
  ${USER_FRAGMENT}
  query GetMe {
    me {
      ...UserFields
    }
  }
`;

export const FIND_MATCHES = gql`
  ${USER_FRAGMENT}
  query FindMatches($userId: ID!, $options: MatchOptions) {
    findMatches(userId: $userId, options: $options) {
      ...UserFields
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_USER = gql`
  ${USER_FRAGMENT}
  mutation CreateUser($input: CreateUserInput!) {
    createUser(input: $input) {
      ...UserFields
    }
  }
`;

export const UPDATE_USER = gql`
  ${USER_FRAGMENT}
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      ...UserFields
    }
  }
`;

export const DELETE_USER = gql`
  mutation DeleteUser($id: ID!) {
    deleteUser(id: $id)
  }
`;
