import { gql } from "graphql-tag";

export const PROFILE_ITEM_FRAGMENT = gql`
  fragment ProfileItemFields on ProfileItem {
    id
    userId
    type
    promptKey
    content
    displayOrder
    createdAt
    updatedAt
  }
`;

export const GET_PROFILE_ITEMS = gql`
  ${PROFILE_ITEM_FRAGMENT}
  query GetProfileItems($userId: ID!) {
    profileItems(userId: $userId) {
      ...ProfileItemFields
    }
  }
`;

export const ADD_PROFILE_ITEM = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation AddProfileItem($input: AddProfileItemInput!) {
    addProfileItem(input: $input) {
      ...ProfileItemFields
    }
  }
`;

export const UPDATE_PROFILE_ITEM = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation UpdateProfileItem($itemId: ID!, $input: UpdateProfileItemInput!) {
    updateProfileItem(itemId: $itemId, input: $input) {
      ...ProfileItemFields
    }
  }
`;

export const DELETE_PROFILE_ITEM = gql`
  mutation DeleteProfileItem($itemId: ID!) {
    deleteProfileItem(itemId: $itemId)
  }
`;

export const REORDER_PROFILE_ITEMS = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation ReorderProfileItems($itemIds: [ID!]!) {
    reorderProfileItems(itemIds: $itemIds) {
      ...ProfileItemFields
    }
  }
`;
