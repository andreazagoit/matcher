import { gql } from "graphql-tag";

// ============================================
// FRAGMENTS
// ============================================

export const USER_FRAGMENT = gql`
  fragment UserFields on User {
    id
    username
    name
    email
    birthdate
    gender
    image
    createdAt
    updatedAt
    sexualOrientation
    heightCm
    relationshipIntent
    relationshipStyle
    hasChildren
    wantsChildren
    religion
    smoking
    drinking
    activityLevel
    jobTitle
    educationLevel
    schoolName
    languages
    ethnicity
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

export const GET_USER = gql`
  ${USER_FRAGMENT}
  query GetUser($username: String!) {
    user(username: $username) {
      ...UserFields
      interests {
        tag
        weight
      }
      profileItems {
        id
        type
        promptKey
        content
        displayOrder
      }
    }
  }
`;

export const GET_USER_WITH_CARDS = gql`
  ${USER_FRAGMENT}
  query GetUserWithCards($username: String!) {
    user(username: $username) {
      ...UserFields
      interests {
        tag
        weight
      }
    }
  }
`;

export const CHECK_USERNAME = gql`
  query CheckUsername($username: String!) {
    checkUsername(username: $username)
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
      interests {
        tag
        weight
      }
    }
  }
`;

export const UPDATE_LOCATION = gql`
  mutation UpdateLocation($lat: Float!, $lon: Float!) {
    updateLocation(lat: $lat, lon: $lon) {
      id
      location {
        lat
        lon
      }
      locationUpdatedAt
    }
  }
`;

