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
    values
    interests
    createdAt
    updatedAt
  }
`;

export const USER_MATCH_FRAGMENT = gql`
  fragment UserMatchFields on User {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    similarity
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
  ${USER_MATCH_FRAGMENT}
  query FindMatches($userId: ID!, $limit: Int) {
    findMatches(userId: $userId, limit: $limit) {
      ...UserMatchFields
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
