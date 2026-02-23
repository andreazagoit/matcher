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
    userItems(userId: $userId) {
      ...ProfileItemFields
    }
  }
`;

export const ADD_PROFILE_ITEM = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation AddUserItem($input: AddUserItemInput!) {
    addUserItem(input: $input) {
      ...ProfileItemFields
    }
  }
`;

export const UPDATE_PROFILE_ITEM = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation UpdateUserItem($itemId: ID!, $input: UpdateUserItemInput!) {
    updateUserItem(itemId: $itemId, input: $input) {
      ...ProfileItemFields
    }
  }
`;

export const DELETE_PROFILE_ITEM = gql`
  mutation DeleteUserItem($itemId: ID!) {
    deleteUserItem(itemId: $itemId)
  }
`;

export const REORDER_PROFILE_ITEMS = gql`
  ${PROFILE_ITEM_FRAGMENT}
  mutation ReorderUserItems($itemIds: [ID!]!) {
    reorderUserItems(itemIds: $itemIds) {
      ...ProfileItemFields
    }
  }
`;
